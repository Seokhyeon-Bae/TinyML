#!/usr/bin/env python3
"""
Deployment-oriented metrics report: FN/FP, missed attacks, false alarm rate.

Integrates baseline eval, optional ratio sweep, and threshold tuning outputs.

Usage:
  python scripts/report_deployment_metrics.py --model models/tflite/saved_model_pruned_qat.tflite
  python scripts/report_deployment_metrics.py --ablation-dir data/processed/ablation/<ts>
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

import yaml

from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown, save_json
from src.data.loader import load_dataset
from src.evaluation.metrics import evaluate_keras_model, evaluate_tflite_model


def _eval_model(path: Path, cfg: dict) -> Dict[str, Any]:
    threshold = float(cfg.get("evaluation", {}).get("prediction_threshold", 0.3))
    data_cfg = cfg.get("data", {})
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    _, _, x_test, y_test = load_dataset(data_cfg.get("name", "cicids2017"), **kwargs)

    if path.suffix == ".h5":
        import tensorflow as tf

        model = tf.keras.models.load_model(str(path), compile=False)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        metrics = evaluate_keras_model(model, x_test, y_test, threshold=threshold)
    elif path.suffix == ".tflite":
        metrics = evaluate_tflite_model(path, x_test, y_test, threshold=threshold)
    else:
        raise ValueError(f"Unsupported model format: {path}")

    tuned_threshold = _tuned_threshold(path, cfg, x_test, y_test)
    if tuned_threshold != threshold:
        if path.suffix == ".h5":
            import tensorflow as tf

            model = tf.keras.models.load_model(str(path), compile=False)
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            tuned = evaluate_keras_model(model, x_test, y_test, threshold=tuned_threshold)
        else:
            tuned = evaluate_tflite_model(path, x_test, y_test, threshold=tuned_threshold)
        metrics["tuned_threshold"] = tuned_threshold
        metrics["tuned_f1"] = tuned["f1"]
        metrics["tuned_attack_recall"] = tuned["attack_recall"]
        metrics["tuned_missed_attacks"] = tuned["missed_attacks"]
        metrics["tuned_false_alarm_rate"] = tuned["false_alarm_rate"]
    return metrics


def _tuned_threshold(path, cfg, x_test, y_test) -> float:
    """Simple F1-max threshold sweep on test set."""
    import numpy as np
    import tensorflow as tf

    if path.suffix == ".h5":
        model = tf.keras.models.load_model(str(path), compile=False)
        y_prob = model.predict(x_test, verbose=0).ravel()
    else:
        interp = tf.lite.Interpreter(model_path=str(path))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        preds = []
        for i in range(len(x_test)):
            interp.set_tensor(inp["index"], x_test[i : i + 1].astype(np.float32))
            interp.invoke()
            preds.append(interp.get_tensor(out["index"])[0])
        y_prob = np.array(preds, dtype=np.float32).ravel()

    best_t, best_f1 = 0.3, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_test == 1) & (y_pred == 1)).sum()
        fp = ((y_test == 0) & (y_pred == 1)).sum()
        fn = ((y_test == 1) & (y_pred == 0)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _from_ablation_dir(ablation_dir: Path, cfg: dict) -> List[Dict[str, Any]]:
    rows = []
    json_path = ablation_dir / "baseline_ablation.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for item in data.get("rows", []):
            rows.append(item)
        return rows

    for model_path in sorted(ablation_dir.rglob("*")):
        if model_path.suffix in {".h5", ".tflite"}:
            m = _eval_model(model_path, cfg)
            m["model_path"] = str(model_path)
            rows.append(m)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Deployment metrics report")
    parser.add_argument("--config", default="config/ablation/fl_baseline.yaml")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--ablation-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(ROOT / args.config)
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "data" / "processed" / "ablation" / "deployment"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    if args.ablation_dir:
        rows = _from_ablation_dir(Path(args.ablation_dir), cfg)
    for model_arg in args.model:
        path = Path(model_arg)
        if path.exists():
            m = _eval_model(path, cfg)
            m["model_path"] = str(path)
            rows.append(m)

    if not rows:
        default = ROOT / "models" / "tflite" / "saved_model_pruned_qat.tflite"
        if default.exists():
            m = _eval_model(default, cfg)
            m["model_path"] = str(default)
            rows.append(m)

    rows_to_csv(rows, out_dir / "deployment_metrics.csv")
    rows_to_markdown(rows, out_dir / "deployment_metrics.md", "Deployment Metrics")
    save_json(rows, out_dir / "deployment_metrics.json")
    print(f"✅ Deployment metrics saved to {out_dir}")


if __name__ == "__main__":
    main()
