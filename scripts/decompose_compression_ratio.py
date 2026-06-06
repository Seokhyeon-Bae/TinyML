#!/usr/bin/env python3
"""
Decompose cumulative compression ratio by pipeline stage (Reviewer B #6).

Usage:
  python scripts/decompose_compression_ratio.py --run-dir data/processed/runs/...
  python scripts/decompose_compression_ratio.py --model models/global_model.h5 --config config/ablation/fl_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown, save_json


def _size_kb(path: Path) -> Optional[float]:
    if path.exists():
        return path.stat().st_size / 1024.0
    return None


def _ratio(base_kb: float, size_kb: Optional[float]) -> Optional[float]:
    if size_kb is None or size_kb <= 0 or base_kb <= 0:
        return None
    return base_kb / size_kb


def decompose_from_paths(
    teacher_path: Path,
    tflite_dir: Path,
    distillation_ratio: float = 2.0,
) -> List[Dict[str, Any]]:
    teacher_kb = _size_kb(teacher_path)
    if teacher_kb is None:
        raise FileNotFoundError(f"Teacher model not found: {teacher_path}")

    stage_files = [
        ("teacher_fl_model", teacher_path),
        ("tflite_original_float", tflite_dir / "saved_model_original.tflite"),
        ("after_pruning_ptq", tflite_dir / "saved_model_qat_ptq.tflite"),
        ("after_pruning_qat", tflite_dir / "saved_model_pruned_qat.tflite"),
        ("traditional_ptq", tflite_dir / "saved_model_no_qat_ptq.tflite"),
        ("traditional_qat", tflite_dir / "saved_model_traditional_qat.tflite"),
    ]

    rows: List[Dict[str, Any]] = []
    cumulative_base = teacher_kb
    for stage, path in stage_files:
        kb = _size_kb(path)
        rows.append(
            {
                "stage": stage,
                "path": str(path),
                "size_kb": round(kb, 3) if kb else "",
                "ratio_vs_teacher": round(_ratio(teacher_kb, kb), 3) if kb else "",
                "step_ratio": round(_ratio(cumulative_base, kb), 3) if kb else "",
            }
        )

    rows.append(
        {
            "stage": "design_note_distillation",
            "path": "",
            "size_kb": "",
            "ratio_vs_teacher": "",
            "step_ratio": f"~{distillation_ratio}x from student width (by architecture design)",
        }
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compression ratio decomposition")
    parser.add_argument("--run-dir", default=None, help="Run directory with models/")
    parser.add_argument("--model", default="models/global_model.h5")
    parser.add_argument("--tflite-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        teacher = run_dir / "models" / "global_model.h5"
        tflite_dir = run_dir / "models" / "tflite"
    else:
        teacher = ROOT / args.model
        tflite_dir = Path(args.tflite_dir) if args.tflite_dir else ROOT / "models" / "tflite"

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "data" / "processed" / "ablation" / "decompose"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = decompose_from_paths(teacher, tflite_dir)
    rows_to_csv(rows, out_dir / "compression_decompose.csv")
    rows_to_markdown(rows, out_dir / "compression_decompose.md", "Compression Ratio Decomposition")
    save_json(rows, out_dir / "compression_decompose.json")
    print(f"✅ Decomposition saved to {out_dir}")


if __name__ == "__main__":
    main()
