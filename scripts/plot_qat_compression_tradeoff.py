#!/usr/bin/env python3
"""
Plot QAT vs PTQ tradeoff from compression sweep CSV (Reviewer B #5).

Usage:
  python scripts/plot_qat_compression_tradeoff.py
  python scripts/plot_qat_compression_tradeoff.py --csv sweep_results_3_4_2026\\ -\\ sweep_results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


def _find_default_csv() -> Path:
    candidates = sorted(ROOT.glob("sweep_results*.csv"))
    candidates += sorted((ROOT / "data" / "processed" / "sweep").rglob("sweep_results.csv"))
    if not candidates:
        raise FileNotFoundError("No sweep_results.csv found")
    return candidates[-1]


def _compression_ratio(row, teacher_kb: float) -> float:
    size = row.get("final_size_kb") or row.get("tflite_size_kb")
    if pd.isna(size) or size <= 0 or teacher_kb <= 0:
        return float("nan")
    return teacher_kb / float(size)


def main():
    parser = argparse.ArgumentParser(description="QAT compression tradeoff plot")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metric", default="final_f1", choices=["final_f1", "final_rec", "final_acc"])
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else _find_default_csv()
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "data" / "processed" / "ablation" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    teacher_kb = float(df["fl_size_kb"].dropna().iloc[0]) if "fl_size_kb" in df.columns else float("nan")
    df["compression_ratio"] = df.apply(lambda r: _compression_ratio(r, teacher_kb), axis=1)

    if "ptq" in df.columns:
        df["quant_mode"] = df["ptq"].map({True: "PTQ", False: "compression_QAT"})
    elif "ptq_no" in df.columns:
        df["quant_mode"] = df["ptq_no"].map({True: "PTQ", False: "compression_QAT"})
    else:
        df["quant_mode"] = "unknown"

    metric_col = args.metric
    if metric_col not in df.columns and "final_f1" in df.columns:
        metric_col = "final_f1"

    plot_df = df.dropna(subset=["compression_ratio", metric_col]).copy()
    summary_path = out_dir / "qat_tradeoff_summary.csv"
    plot_df[["tag", "compression_ratio", metric_col, "quant_mode", "final_size_kb"]].to_csv(
        summary_path, index=False
    )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for mode, group in plot_df.groupby("quant_mode"):
            ax.scatter(
                group["compression_ratio"],
                group[metric_col],
                label=str(mode),
                alpha=0.7,
            )
        ax.set_xlabel("Compression ratio (teacher KB / final KB)")
        ax.set_ylabel(metric_col)
        ax.set_title("Accuracy vs compression ratio (QAT vs PTQ)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        png_path = out_dir / "qat_compression_tradeoff.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"✅ Plot saved: {png_path}")
    except ImportError:
        print("⚠️  matplotlib not installed; CSV summary only")

    print(f"✅ Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
