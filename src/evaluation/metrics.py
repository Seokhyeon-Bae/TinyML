"""Binary classification metrics with deployment-oriented counts (FN/FP)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, F1, attack recall, and confusion counts."""
    y_true = y_true.astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float32).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n = max(len(y_true), 1)
    accuracy = (tp + tn) / n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    attack_recall = recall
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    false_alarm_rate = fp / max(fp + tn, 1)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "attack_recall": float(attack_recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "missed_attacks": fn,
        "false_alarms": fp,
        "false_alarm_rate": float(false_alarm_rate),
        "threshold": float(threshold),
    }


def evaluate_keras_model(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_prob = model.predict(x_test, verbose=0)
    if y_prob.ndim == 2:
        if y_prob.shape[1] == 1:
            y_prob = y_prob.ravel()
        elif y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
    metrics = compute_binary_metrics(y_test, y_prob, threshold=threshold)
    metrics["model_type"] = "keras"
    return metrics


def evaluate_tflite_model(
    tflite_path: str | Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    latency_runs: int = 100,
) -> Dict[str, Any]:
    import tensorflow as tf

    path = Path(tflite_path)
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    preds = []
    for i in range(len(x_test)):
        sample = x_test[i : i + 1].astype(np.float32)
        interp.set_tensor(inp["index"], sample)
        interp.invoke()
        preds.append(interp.get_tensor(out["index"])[0])

    y_prob = np.array(preds, dtype=np.float32)
    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        y_prob = y_prob.ravel()
    elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]

    metrics = compute_binary_metrics(y_test, y_prob, threshold=threshold)
    metrics["model_type"] = "tflite"
    metrics["size_kb"] = path.stat().st_size / 1024.0
    metrics["latency_ms"] = measure_tflite_latency_ms(
        str(path), x_test, runs=latency_runs
    )
    return metrics


def measure_tflite_latency_ms(
    tflite_path: str,
    x_test: np.ndarray,
    runs: int = 100,
) -> float:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    sample = x_test[0:1].astype(np.float32)

    # Warmup
    for _ in range(5):
        interp.set_tensor(inp["index"], sample)
        interp.invoke()

    start = time.perf_counter()
    n = min(runs, len(x_test))
    for i in range(n):
        interp.set_tensor(inp["index"], x_test[i : i + 1].astype(np.float32))
        interp.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / max(n, 1)
