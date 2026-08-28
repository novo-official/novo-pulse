"""Local LLM narrator, tested against a real HTTP server.

Ollama itself is not installed in CI, but the narrator only ever speaks its
HTTP API - so serving that API faithfully exercises everything on our side:
the request shape, the response parsing, `<think>` stripping, and above all the
anti-hallucination gate. A mocked `urlopen` would prove none of that.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ml.insights.narrator import LocalLLMNarrator, TemplateNarrator, get_narrator

FACTS = {
    "label": "کیش",
    "change_pct": 18.2,
    "forecast_total": 14500,
    "horizon": 30,
    "drivers": [{"group": "holiday", "label_fa": "تعطیلات", "direction": "positive"}],
    "confidence": {"lower": 12000, "upper": 17800, "label": "medium"},
}

REPLIES = {
    # Uses only numbers it was given.
    "grounded": "تقاضای کیش طی ۳۰ روز آینده حدود ۱۸.۲ درصد افزایش می‌یابد و به ۱۴۵۰۰ واحد می‌رسد.",
    # Invents both a percentage and a level - must be rejected.
    "ungrounded": "تقاضا ۹۹ درصد رشد می‌کند و به ۸۷۶۵۴۳ واحد می‌رسد.",
    # A reasoning model's scratchpad must not reach the user.
    "thinking": "<think>internal reasoning here</think>تقاضا ۱۸.۲ درصد افزایش می‌یابد.",
    "empty": "",
}


class _Handler(BaseHTTPRequestHandler):
    received: list[dict] = []

    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": f"{name}:latest"} for name in REPLIES]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        _Handler.received.append(request)
        model = str(request.get("model", "grounded")).split(":")[0]
        if model == "broken":
            self._send(200, b"{not valid json")
            return
        self._send(200, {"response": REPLIES.get(model, REPLIES["grounded"]), "done": True})


@pytest.fixture(scope="module")
def ollama():
    """A throwaway Ollama-compatible server on an ephemeral port."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(autouse=True)
def _clear():
    _Handler.received.clear()


# ------------------------------------------------------------- availability
def test_narrator_detects_a_reachable_server(ollama, monkeypatch):
    monkeypatch.setenv("ENABLE_LOCAL_LLM", "true")
    assert LocalLLMNarrator(base_url=ollama, model="grounded").available is True


def test_narrator_is_unavailable_without_the_feature_flag(ollama, monkeypatch):
    monkeypatch.delenv("ENABLE_LOCAL_LLM", raising=False)
    assert LocalLLMNarrator(base_url=ollama, model="grounded").available is False


def test_get_narrator_picks_the_llm_when_it_is_reachable(ollama, monkeypatch):
    monkeypatch.setenv("ENABLE_LOCAL_LLM", "true")
    monkeypatch.setenv("OLLAMA_BASE_URL", ollama)
    monkeypatch.setenv("OLLAMA_MODEL", "grounded")
    assert isinstance(get_narrator(), LocalLLMNarrator)


def test_get_narrator_falls_back_to_the_template_when_it_is_not(monkeypatch):
    monkeypatch.setenv("ENABLE_LOCAL_LLM", "true")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:59999")
    assert isinstance(get_narrator(), TemplateNarrator)


# ------------------------------------------------------------- request shape
def test_the_request_carries_the_anti_hallucination_prompt_and_the_statistics(
    ollama, monkeypatch
):
    monkeypatch.setenv("ENABLE_LOCAL_LLM", "true")
    LocalLLMNarrator(base_url=ollama, model="grounded").narrate(FACTS)

    assert _Handler.received, "the narrator never called the server"
    sent = _Handler.received[-1]
    prompt = sent["prompt"]

    assert "Never invent causes, numbers, events or business facts" in prompt
    assert '"forecast_total": 14500' in prompt
    assert sent["stream"] is False, "a streamed reply cannot be validated before use"
    assert sent["options"]["temperature"] <= 0.3, "narration must be near-deterministic"


# --------------------------------------------------------------- behaviour
def test_a_grounded_reply_is_accepted(ollama, monkeypatch):
    monkeypatch.setenv("ENABLE_LOCAL_LLM", "true")
    result = LocalLLMNarrator(base_url=ollama, model="grounded").narrate(FACTS)

    assert result["source"].startswith("ollama")
    assert "کیش" in result["text"]
    assert result["grounded"] is True


def test_an_invented_number_is_rejected_and_never_reaches_the_user(ollama, monkeypatch):
    """The whole point of the gate: a hallucinated figure must not be shown."""
    monkeypatch.setenv("ENABLE_LOCAL_LLM", "true")
    result = LocalLLMNarrator(base_url=ollama, model="ungrounded").narrate(FACTS)

    assert "template" in result["source"], "the ungrounded reply should have been discarded"
    assert "876543" not in result["text"]
    assert "۸۷۶۵۴۳" not in result["text"]
    assert "۹۹" not in result["text"]
    assert len(result["text"]) > 30, "the fallback must still say something useful"


def test_reasoning_scratchpad_is_stripped(ollama, monkeypatch):
    monkeypatch.setenv("ENABLE_LOCAL_LLM", "true")
    result = LocalLLMNarrator(base_url=ollama, model="thinking").narrate(FACTS)

    assert "<think>" not in result["text"]
    assert "internal reasoning" not in result["text"]


@pytest.mark.parametrize("model", ["empty", "broken"])
def test_a_useless_reply_falls_back_to_the_template(ollama, monkeypatch, model):
    monkeypatch.setenv("ENABLE_LOCAL_LLM", "true")
    result = LocalLLMNarrator(base_url=ollama, model=model).narrate(FACTS)

    assert "template" in result["source"]
    assert len(result["text"]) > 30


def test_an_unreachable_server_falls_back_without_raising(monkeypatch):
    monkeypatch.setenv("ENABLE_LOCAL_LLM", "true")
    result = LocalLLMNarrator(base_url="http://127.0.0.1:59999", model="grounded").narrate(FACTS)

    assert "template" in result["source"]
    assert len(result["text"]) > 30


def test_persian_and_arabic_digits_are_validated_too(ollama, monkeypatch):
    """A model answering in Persian numerals must be checked, not waved through."""
    monkeypatch.setenv("ENABLE_LOCAL_LLM", "true")
    narrator = LocalLLMNarrator(base_url=ollama, model="grounded")

    assert narrator._validate("رشد ۱۸.۲ درصدی به ۱۴۵۰۰ واحد", FACTS) is True
    assert narrator._validate("رشد ٤٢ درصدی", FACTS) is False


def test_narration_never_raises_whatever_the_server_does(ollama, monkeypatch):
    """Narration is decoration; it must never be able to fail a request."""
    monkeypatch.setenv("ENABLE_LOCAL_LLM", "true")
    for model in (*REPLIES, "broken", "unknown-model"):
        result = LocalLLMNarrator(base_url=ollama, model=model).narrate(FACTS)
        assert result["text"], f"{model} produced no text at all"
