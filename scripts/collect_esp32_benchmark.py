#!/usr/bin/env python3
"""
Parse ESP32 Serial benchmark logs into JSON.

Expected Serial lines:
  BENCHMARK latency_us=1234 arena_used=5678 input_dim=78
  BENCHMARK latency_us=... (multiple runs)

Usage:
  python scripts/collect_esp32_benchmark.py --port /dev/cu.usbserial-* --runs 50
  python scripts/collect_esp32_benchmark.py --log-file esp32_serial.log
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent


def parse_benchmark_lines(text: str) -> List[dict]:
    rows = []
    pattern = re.compile(
        r"latency_us=(?P<latency>\d+(?:\.\d+)?)"
        r"(?:\s+arena_used=(?P<arena>\d+))?"
        r"(?:\s+input_dim=(?P<input_dim>\d+))?"
    )
    for line in text.splitlines():
        if "BENCHMARK" not in line:
            continue
        m = pattern.search(line)
        if m:
            row = {"latency_us": float(m.group("latency"))}
            if m.group("arena"):
                row["arena_used"] = int(m.group("arena"))
            if m.group("input_dim"):
                row["input_dim"] = int(m.group("input_dim"))
            rows.append(row)
    return rows


def summarize(rows: List[dict]) -> dict:
    if not rows:
        return {"count": 0}
    latencies = [r["latency_us"] for r in rows]
    summary = {
        "count": len(rows),
        "latency_us_mean": mean(latencies),
        "latency_us_median": median(latencies),
        "latency_ms_mean": mean(latencies) / 1000.0,
        "arena_used": rows[-1].get("arena_used"),
        "input_dim": rows[-1].get("input_dim"),
    }
    return summary


def read_serial(port: str, timeout_s: float, runs: int) -> str:
    try:
        import serial
    except ImportError as e:
        raise ImportError("pip install pyserial") from e

    buf = []
    with serial.Serial(port, 115200, timeout=1) as ser:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if line:
                buf.append(line)
                print(line)
            if len(parse_benchmark_lines("\n".join(buf))) >= runs:
                break
    return "\n".join(buf)


def main():
    parser = argparse.ArgumentParser(description="Collect ESP32 benchmark from Serial")
    parser.add_argument("--port", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--output",
        default="data/processed/ablation/esp32_benchmark.json",
    )
    args = parser.parse_args()

    if args.log_file:
        text = Path(args.log_file).read_text(encoding="utf-8")
    elif args.port:
        text = read_serial(args.port, args.timeout, args.runs)
    else:
        print("Provide --port or --log-file", file=sys.stderr)
        return 1

    rows = parse_benchmark_lines(text)
    result = {"samples": rows, "summary": summarize(rows)}
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"✅ ESP32 benchmark saved: {out}")
    if result["summary"].get("latency_ms_mean"):
        print(f"   Mean latency: {result['summary']['latency_ms_mean']:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
