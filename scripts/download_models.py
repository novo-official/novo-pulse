#!/usr/bin/env python3
"""Pre-fetch optional model weights so the system works offline.

Competition venues have unreliable internet. Run this once while you still
have a connection; afterwards set `HF_HUB_OFFLINE=1` and everything loads from
the local cache.

    python scripts/download_models.py                # everything available
    python scripts/download_models.py --only chronos
    python scripts/download_models.py --check        # report, download nothing

Nothing here is required: if a download fails, Novo Pulse still runs on
LightGBM, CatBoost and the statistical baselines.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

DEFAULT_CACHE = REPO_ROOT / "models" / "hf_cache"

TARGETS = {
    "chronos": {
        "label": "Chronos-2 (zero-shot foundation forecasting)",
        "env": "CHRONOS_MODEL_ID",
        "default_id": "amazon/chronos-2",
        # Tried in order - the first one that downloads wins.
        "fallbacks": ["amazon/chronos-bolt-base", "amazon/chronos-t5-small"],
        "requires": ["torch", "chronos"],
        "enable_flag": "ENABLE_CHRONOS",
    },
    "llm": {
        "label": "Qwen3-4B via Ollama (insight narration only)",
        "env": "OLLAMA_MODEL",
        "default_id": "qwen3:4b",
        "fallbacks": [],
        "requires": [],
        "enable_flag": "ENABLE_LOCAL_LLM",
        "ollama": True,
    },
}


def _ok(message: str) -> None:
    print(f"  \033[32mOK\033[0m    {message}")


def _skip(message: str) -> None:
    print(f"  \033[33mSKIP\033[0m  {message}")


def _fail(message: str) -> None:
    print(f"  \033[31mFAIL\033[0m  {message}")


def check_imports(modules: list[str]) -> tuple[bool, str]:
    import importlib.util

    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        return False, f"missing packages: {', '.join(missing)} (pip install -r requirements-optional.txt)"
    return True, "ready"


def download_chronos(check_only: bool) -> bool:
    spec = TARGETS["chronos"]
    available, reason = check_imports(spec["requires"])
    if not available:
        _skip(f"{spec['label']} - {reason}")
        return False
    if check_only:
        _ok(f"{spec['label']} - dependencies present")
        return True

    from chronos import BaseChronosPipeline

    candidates = [os.getenv(spec["env"], spec["default_id"]), *spec["fallbacks"]]
    for model_id in candidates:
        try:
            print(f"        fetching {model_id} ...")
            BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")
            _ok(f"{spec['label']} - cached '{model_id}'")
            print(f"        set CHRONOS_MODEL_ID={model_id} and ENABLE_CHRONOS=true")
            return True
        except Exception as exc:  # noqa: BLE001 - report and try the next one
            _fail(f"{model_id}: {type(exc).__name__}: {str(exc)[:140]}")
    return False


def download_neuralforecast(check_only: bool) -> bool:
    available, reason = check_imports(["neuralforecast", "torch"])
    if not available:
        _skip(f"NHITS / NBEATSx - {reason}")
        return False
    _ok("NHITS / NBEATSx - no pre-trained weights needed (trained from scratch)")
    if not check_only:
        print("        set ENABLE_NEURALFORECAST=true to include them")
    return True


def download_ollama(check_only: bool) -> bool:
    import json
    import urllib.request

    spec = TARGETS["llm"]
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv(spec["env"], spec["default_id"])
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=4) as response:
            tags = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - Ollama is entirely optional
        _skip(
            f"{spec['label']} - Ollama is not reachable at {base}. "
            "The template narrator will be used instead."
        )
        return False

    installed = {entry.get("name", "") for entry in tags.get("models", [])}
    if any(name.startswith(model.split(":")[0]) for name in installed):
        _ok(f"{spec['label']} - '{model}' already pulled")
        return True
    if check_only:
        _skip(f"{spec['label']} - not pulled yet. Run: ollama pull {model}")
        return False

    print(f"        pulling {model} via Ollama (this can take a few minutes) ...")
    request = urllib.request.Request(
        f"{base}/api/pull",
        data=json.dumps({"name": model, "stream": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            response.read()
        _ok(f"{spec['label']} - pulled '{model}'")
        return True
    except Exception as exc:  # noqa: BLE001
        _fail(f"could not pull {model}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["chronos", "neuralforecast", "llm"],
        help="Download just one component.",
    )
    parser.add_argument("--check", action="store_true", help="Report status without downloading.")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory.")
    args = parser.parse_args()

    cache = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))

    print("Novo Pulse - optional model setup")
    print(f"  cache directory: {cache}")
    print("  (every component below is optional; the system runs without all of them)\n")

    results: dict[str, bool] = {}
    if args.only in (None, "chronos"):
        results["chronos"] = download_chronos(args.check)
    if args.only in (None, "neuralforecast"):
        results["neuralforecast"] = download_neuralforecast(args.check)
    if args.only in (None, "llm"):
        results["llm"] = download_ollama(args.check)

    print("\nSummary")
    for name, ok in results.items():
        print(f"  {name:<16} {'available' if ok else 'not available (a fallback will be used)'}")
    print("\nFor a fully offline run afterwards, export:")
    print(f"  export HF_HOME={cache}")
    print("  export HF_HUB_OFFLINE=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
