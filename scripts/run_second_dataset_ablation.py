#!/usr/bin/env python3
"""
Second-dataset generalization ablation: (b) FL + (c) compression on Bot-IoT or UNSW-NB15.

Usage:
  python scripts/run_second_dataset_ablation.py --dataset bot_iot
  python scripts/run_second_dataset_ablation.py --dataset unsw_nb15 --quick
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablation_utils import (
    apply_quick_mode,
    load_yaml,
    make_output_dir,
    rows_to_csv,
    rows_to_markdown,
    run_cmd,
    save_json,
    save_yaml,
)
from scripts.run_baseline_ablation import (
    _evaluate_keras_path,
    _evaluate_tflite_path,
    _run_compression,
)


def main():
    parser = argparse.ArgumentParser(description="Second dataset ablation (P2)")
    parser.add_argument("--dataset", default="bot_iot", choices=["bot_iot", "unsw_nb15"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else make_output_dir(f"ablation/{args.dataset}")
    models_dir = out_dir / "models"
    tflite_dir = out_dir / "tflite"
    configs_dir = out_dir / "configs"
    for d in (models_dir, tflite_dir, configs_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg = apply_quick_mode(load_yaml(ROOT / "config" / "ablation" / "fl_baseline.yaml"), args.quick)
    cfg["data"]["name"] = args.dataset
    if args.dataset == "bot_iot":
        cfg["data"]["path"] = "data/raw/Bot-IoT"
    else:
        cfg["data"]["path"] = "data/raw/UNSW-NB15"

    cfg_path = configs_dir / f"{args.dataset}_fl.yaml"
    save_yaml(cfg_path, cfg)

    model_b = models_dir / f"{args.dataset}_fl.h5"
    rows = []

    if not args.skip_train:
        run_cmd(
            [
                sys.executable,
                "-m",
                "src.federated.client",
                "--config",
                str(cfg_path),
                "--save-model",
                str(model_b),
            ],
            f"FL on {args.dataset}",
        )

    if model_b.exists():
        rows.append(_evaluate_keras_path(model_b, cfg, "b", f"FL+{args.dataset}"))

    if model_b.exists():
        trad = models_dir / f"{args.dataset}_traditional.h5"
        _run_compression(cfg_path, model_b, tflite_dir, trad)
        ptq = tflite_dir / "saved_model_no_qat_ptq.tflite"
        qat = tflite_dir / "saved_model_traditional_qat.tflite"
        alt_ptq = tflite_dir / "saved_model_qat_ptq.tflite"
        if ptq.exists():
            rows.append(_evaluate_tflite_path(ptq, cfg, "c", f"FL+PTQ+{args.dataset}"))
        elif alt_ptq.exists():
            rows.append(_evaluate_tflite_path(alt_ptq, cfg, "c", f"FL+PTQ+{args.dataset}"))
        if qat.exists():
            rows.append(_evaluate_tflite_path(qat, cfg, "d", f"FL+QAT+{args.dataset}"))

    rows_to_csv(rows, out_dir / f"{args.dataset}_ablation.csv")
    rows_to_markdown(rows, out_dir / f"{args.dataset}_ablation.md", f"Second Dataset Ablation ({args.dataset})")
    save_json(rows, out_dir / f"{args.dataset}_ablation.json")
    print(f"✅ Second dataset ablation saved to {out_dir}")


if __name__ == "__main__":
    main()
