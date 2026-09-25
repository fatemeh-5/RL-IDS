#!/usr/bin/env python3
"""Scan experiments/MULTISEED/runs/**/METRICS_ROW.json into aggregate_results.csv.

Safe to run at any point during the sweep (partial results are fine) and to
re-run repeatedly as more cells finish — it always rebuilds the CSV fresh
from whatever METRICS_ROW.json files exist on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "experiments" / "MULTISEED" / "runs"
OUT_CSV = ROOT / "experiments" / "MULTISEED" / "aggregate_results.csv"

COLUMNS = [
    "config", "feature_set", "model_type", "seed",
    "KT_Accuracy", "KT_Precision", "KT_Recall", "KT_F1", "KT_ROC_AUC",
    "Weighted_ZeroDay", "Macro_ZeroDay",
    "Shellcode", "Brute Force", "Theft", "ransomware", "Backdoor",
]


def main() -> None:
    rows = []
    for marker in sorted(RUNS_ROOT.glob("*/*/*/*/METRICS_ROW.json")):
        with marker.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))

    if not rows:
        print(f"No METRICS_ROW.json files found under {RUNS_ROOT}", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Rows missing expected columns: {missing}")
    df = df[COLUMNS].sort_values(["model_type", "config", "feature_set", "seed"]).reset_index(drop=True)

    dupes = df.duplicated(subset=["config", "feature_set", "model_type", "seed"], keep=False)
    if dupes.any():
        print("WARNING: duplicate (config, feature_set, model_type, seed) rows found:", file=sys.stderr)
        print(df[dupes], file=sys.stderr)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_CSV}")

    counts = df.groupby(["model_type", "config", "feature_set"]).size().rename("n_seeds")
    print("\nSeed coverage per (model_type, config, feature_set):")
    print(counts.to_string())


if __name__ == "__main__":
    main()
