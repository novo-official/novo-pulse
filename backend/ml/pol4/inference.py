"""Run Pol 4 inference from a verified model bundle without fitting anything.

Example:

    PYTHONPATH=backend python -m ml.pol4.inference --variant calibrated
    PYTHONPATH=backend python -m ml.pol4.inference --variant raw
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .artifacts import build_input_manifest
from .champion import FittedChampion
from .config import Pol4Config
from .loader import load_pol4
from .submission import (
    build_named_submission,
    build_submission,
    validate_submission,
    write_submission,
)


MODEL_VARIANTS = {
    "raw": Path("model_variants/raw"),
    "calibrated": Path("model_variants/calibrated"),
}


def run_from_bundle(
    config: Pol4Config | None = None,
    *,
    model_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    variant: str = "calibrated",
) -> dict[str, Any]:
    """Verify provenance, load native boosters and write a valid submission."""
    config = config or Pol4Config()
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown model variant {variant!r}; choose from {sorted(MODEL_VARIANTS)}")
    model_dir = Path(model_dir or config.artifacts_dir / MODEL_VARIANTS[variant])
    output_path = Path(
        output_path or config.artifacts_dir / f"results_{variant}_from_bundle.csv"
    )

    data = load_pol4(config)
    inputs = build_input_manifest(config, data)
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"model manifest is missing: {manifest_path}")
    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if model_manifest.get("input_digest") != inputs["input_digest"]:
        raise ValueError("model bundle was built from different input CSVs")

    fitted = FittedChampion.load(model_dir)
    predictions = fitted.predict(config.target_dates(), data)
    submission = build_submission(predictions)
    report = validate_submission(submission, data.city_codes, config, raise_on_error=True)
    write_submission(submission, output_path)
    named_path = output_path.with_name(f"{output_path.stem}_named{output_path.suffix}")
    write_submission(build_named_submission(predictions, data), named_path)
    return {
        "valid": report.valid,
        "rows": report.rows,
        "variant": variant,
        "model_bundle": str(model_dir),
        "bundle_digest": model_manifest["bundle_digest"],
        "input_digest": inputs["input_digest"],
        "output": str(output_path),
        "named_output": str(named_path),
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = Pol4Config()
    parser = argparse.ArgumentParser(
        prog="python -m ml.pol4.inference",
        description="Pol 4 inference from a checksum-verified native model bundle",
    )
    parser.add_argument("--raw-dir", type=Path, default=defaults.raw_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=defaults.artifacts_dir)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--variant",
        choices=tuple(MODEL_VARIANTS),
        default="calibrated",
        help="saved champion variant; --model-dir may override its default directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = replace(
        Pol4Config(), raw_dir=args.raw_dir, artifacts_dir=args.artifacts_dir
    )
    result = run_from_bundle(
        config,
        model_dir=args.model_dir,
        output_path=args.output,
        variant=args.variant,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
