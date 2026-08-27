"""Turning structured insights into Persian prose.

Hard rule: the narrator never computes or invents a number. It receives a
closed JSON structure and renders it. Two implementations:

* `TemplateNarrator`   - always available, fully deterministic.
* `LocalLLMNarrator`   - optional, talks to a local Ollama instance. If Ollama
  is not running, or the response fails validation, we fall back to templates.

No paid or cloud API is used anywhere.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a demand-forecasting analyst writing in Persian (Farsi).\n"
    "Only explain the supplied statistics.\n"
    "Never invent causes, numbers, events or business facts.\n"
    "Never state a causal claim; describe associations only.\n"
    "If the evidence is insufficient, say so explicitly.\n"
    "Answer in 2-4 short Persian sentences. Do not use markdown."
)


def _fmt(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "نامشخص"
    return f"{value:,.{digits}f}"


class TemplateNarrator:
    """Deterministic Persian narrative built from the structured insights."""

    name = "template"
    available = True

    def narrate(self, payload: dict[str, Any]) -> dict[str, Any]:
        sentences: list[str] = []

        market = payload.get("market") or {}
        horizon = payload.get("horizon") or market.get("horizon") or 30
        entity_label = payload.get("label") or payload.get("entity_label")
        change = payload.get("change_pct", market.get("change_pct"))
        total = payload.get("forecast_total", market.get("total_forecast"))

        subject = entity_label or "کل بازار"
        if change is None:
            sentences.append(
                f"تقاضای پیش‌بینی‌شده {subject} طی {horizon} روز آینده "
                f"{_fmt(total)} واحد است."
            )
        else:
            direction = "افزایش" if change >= 0 else "کاهش"
            sentences.append(
                f"تقاضای {subject} طی {horizon} روز آینده حدود {abs(change):.1f}٪ "
                f"{direction} خواهد یافت و به {_fmt(total)} واحد می‌رسد."
            )

        drivers = payload.get("drivers") or []
        positive = [d for d in drivers if d.get("direction") == "positive"][:2]
        negative = [d for d in drivers if d.get("direction") == "negative"][:2]
        if positive or negative:
            parts = []
            if positive:
                parts.append("عوامل افزایشی: " + "، ".join(d.get("label_fa", d["group"]) for d in positive))
            if negative:
                parts.append("عوامل کاهشی: " + "، ".join(d.get("label_fa", d["group"]) for d in negative))
            sentences.append("؛ ".join(parts) + ".")
        elif drivers:
            names = "، ".join(d.get("label_fa", d["group"]) for d in drivers[:3])
            sentences.append(f"مهم‌ترین عوامل مؤثر بر این پیش‌بینی: {names}.")

        confidence = payload.get("confidence") or {}
        if confidence.get("lower") is not None and confidence.get("upper") is not None:
            sentences.append(
                f"بازه اطمینان ۸۰٪ بین {_fmt(confidence['lower'])} و "
                f"{_fmt(confidence['upper'])} واحد قرار دارد."
            )
        label = confidence.get("label")
        if label:
            mapping = {"high": "بالا", "medium": "متوسط", "low": "پایین"}
            sentences.append(f"سطح اطمینان مدل برای این بازه {mapping.get(label, label)} است.")

        model = payload.get("model_confidence") or {}
        if model.get("improvement_pct") is not None:
            sentences.append(
                f"مدل منتخب ({model.get('champion')}) در اعتبارسنجی "
                f"{model['improvement_pct']:.1f}٪ بهتر از بهترین مدل پایه عمل کرده است."
            )

        if not sentences:
            sentences.append("داده کافی برای تولید تحلیل در دسترس نیست.")

        return {"text": " ".join(sentences), "source": self.name, "grounded": True}


class LocalLLMNarrator:
    """Optional Ollama narrator. Structured JSON in, Persian prose out."""

    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: int = 45):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b")
        self.timeout = timeout
        self.fallback = TemplateNarrator()

    @property
    def available(self) -> bool:
        if os.getenv("ENABLE_LOCAL_LLM", "false").lower() not in {"1", "true", "yes"}:
            return False
        try:
            import urllib.request

            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as response:
                return response.status == 200
        except Exception:  # noqa: BLE001 - unreachable Ollama is a normal state
            return False

    def narrate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.available:
            result = self.fallback.narrate(payload)
            result["source"] = "template (local LLM unavailable)"
            return result
        try:
            text = self._call(payload)
            if not text or not self._validate(text, payload):
                raise ValueError("response failed grounding validation")
            return {"text": text.strip(), "source": f"{self.name}:{self.model}", "grounded": True}
        except Exception as exc:  # noqa: BLE001
            log.warning("Local LLM narration failed (%s); using the template narrator", exc)
            result = self.fallback.narrate(payload)
            result["source"] = "template (LLM fallback)"
            return result

    def _call(self, payload: dict[str, Any]) -> str:
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                "prompt": (
                    f"{SYSTEM_PROMPT}\n\nSTATISTICS (JSON):\n"
                    f"{json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
                    "Persian summary:"
                ),
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 300},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return _strip_thinking(data.get("response", ""))

    def _validate(self, text: str, payload: dict[str, Any]) -> bool:
        """Reject any number the model did not receive.

        Every numeric token in the response must appear in the supplied
        statistics (to within rounding). This is the anti-hallucination gate.
        """
        allowed = _numeric_tokens(payload)
        for token in re.findall(r"\d+(?:[.,]\d+)?", _to_latin_digits(text)):
            value = float(token.replace(",", ""))
            if not any(abs(value - candidate) <= max(1.0, abs(candidate) * 0.02) for candidate in allowed):
                log.warning("LLM produced ungrounded number %s", value)
                return False
        return True


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _to_latin_digits(text: str) -> str:
    persian = "۰۱۲۳۴۵۶۷۸۹"
    arabic = "٠١٢٣٤٥٦٧٨٩"
    for index, digit in enumerate(persian):
        text = text.replace(digit, str(index))
    for index, digit in enumerate(arabic):
        text = text.replace(digit, str(index))
    return text


def _numeric_tokens(payload: Any, seen: set[float] | None = None) -> set[float]:
    seen = seen if seen is not None else set()
    if isinstance(payload, dict):
        for value in payload.values():
            _numeric_tokens(value, seen)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _numeric_tokens(value, seen)
    elif isinstance(payload, bool):
        pass
    elif isinstance(payload, (int, float)):
        value = float(payload)
        seen.update({value, abs(value), round(value), round(value, 1), abs(value) * 100})
    return seen


def get_narrator() -> TemplateNarrator | LocalLLMNarrator:
    """Pick the best narrator that is actually usable right now."""
    if os.getenv("ENABLE_LOCAL_LLM", "false").lower() in {"1", "true", "yes"}:
        narrator = LocalLLMNarrator()
        if narrator.available:
            return narrator
    return TemplateNarrator()
