#!/usr/bin/env python3
"""
Run PGD + FGSM evaluation on ablation TFLite outputs (Reviewer B #4).

Usage:
  python scripts/run_pgd_ablation.py --ablation-dir data/processed/ablation/<timestamp>
  python scripts/run_pgd_ablation.py --models models/tflite/a.tflite models/tflite/b.tflite --keras models/global_model.h5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_baseline_ablation import _run_pgd_on_models


def main():
    parser = argparse.ArgumentParser(description="PGD/FGSM ablation eval")
    parser.add_argument("--ablation-dir", default=None)
    parser.add_argument("--config", default="config/ablation/fl_baseline.yaml")
    parser.add_argument("--keras", default=None, help="Keras .h5 for adversarial example generation")
    parser.add_argument("--models", nargs="*", default=[])
    args = parser.parse_args()

    cfg = ROOT / args.config
    models = [Path(p) for p in args.models]

    if args.ablation_dir:
        ab_dir = Path(args.ablation_dir)
        tflite_dir = ab_dir / "tflite"
        models_dir = ab_dir / "models"
        keras = models_dir / "b_fl.h5"
        if keras.exists():
            models = [keras] + sorted(tflite_dir.glob("*.tflite"))
        out = ab_dir / "pgd"
    else:
        out = ROOT / "data" / "processed" / "ablation" / "pgd"
        if args.keras:
            models = [Path(args.keras)] + models

    if not models:
        print("No models found.", file=sys.stderr)
        return 1

    _run_pgd_on_models(cfg, models, out / "pgd", attack="pgd")
    _run_pgd_on_models(cfg, models, out / "fgsm", attack="fgsm")
    print(f"✅ PGD/FGSM ablation saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
