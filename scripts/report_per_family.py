#!/usr/bin/env python3
"""Per-attack-family report: mean +/- std Precision/Recall/F1 (and Detection
Rate) across seeds, for every (config, feature_set) x (known attack class OR
zero-day family) combination.

This is the ONE script that produces the per-family breakdown for the paper
(Phase 4). It reads only experiments/MULTISEED/runs/**/METRICS_ROW.json (for
which cells exist) plus each cell's family_breakdown_precision_recall_f1.csv
(written by agent.evaluation.metrics.evaluate_family_breakdown / layout.py),
never a machine-specific path.

Safe to run at any point during the sweep (partial results are fine) and to
re-run repeatedly as more seeds finish.

Outputs:
  experiments/MULTISEED/per_family_precision_recall_f1.csv   (long, all seeds)
  experiments/MULTISEED/per_family_summary.csv                (mean/std/n)
  experiments/MULTISEED/per_family_summary.json
  Appends a dated, versioned section to paper_results_log.md
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "experiments" / "MULTISEED" / "runs"
OUT_LONG = ROOT / "experiments" / "MULTISEED" / "per_family_precision_recall_f1.csv"
OUT_SUMMARY_CSV = ROOT / "experiments" / "MULTISEED" / "per_family_summary.csv"
OUT_SUMMARY_JSON = ROOT / "experiments" / "MULTISEED" / "per_family_summary.json"
PAPER_LOG = ROOT / "paper_results_log.md"

ZERO_DAY_FAMILY_ORDER = ["Shellcode", "Brute Force", "Theft", "ransomware", "Backdoor"]
MODEL_ORDER = [
    "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11",
    "RF", "XGB", "MLP", "LSTM",
]


def _family_breakdown_path(metrics_row_path: Path) -> Path | None:
    """DRL cells write to <run_dir>/results/family_breakdown_....csv;
    supervised cells write directly to <run_dir>/family_breakdown_....csv."""
    run_dir = metrics_row_path.parent
    candidates = [
        run_dir / "results" / "family_breakdown_precision_recall_f1.csv",
        run_dir / "family_breakdown_precision_recall_f1.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_long() -> pd.DataFrame:
    rows = []
    metrics_files = sorted(RUNS_ROOT.glob("*/*/*/*/METRICS_ROW.json"))
    missing = 0
    for metrics_path in metrics_files:
        with metrics_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        fam_path = _family_breakdown_path(metrics_path)
        if fam_path is None:
            missing += 1
            continue
        fam_df = pd.read_csv(fam_path)
        fam_df.insert(0, "Seed", meta["seed"])
        fam_df.insert(0, "ModelType", meta["model_type"])
        fam_df.insert(0, "FeatureSet", meta["feature_set"])
        fam_df.insert(0, "Config", meta["config"])
        rows.append(fam_df)

    if missing:
        print(f"[WARN] {missing} cell(s) had a METRICS_ROW.json but no "
              f"family_breakdown_precision_recall_f1.csv (older schema run?).",
              file=sys.stderr)
    if not rows:
        raise SystemExit(
            f"No family_breakdown_precision_recall_f1.csv files found under {RUNS_ROOT}. "
            "Has the sweep produced any completed cells yet?"
        )
    return pd.concat(rows, ignore_index=True)


def build_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Detection Rate (%)", "Precision (%)", "Recall (%)", "F1 (%)"]
    group_cols = ["Config", "FeatureSet", "ModelType", "Kind", "Attack"]
    rows = []
    for keys, group in long_df.groupby(group_cols, sort=True):
        config, feature_set, model_type, kind, attack = keys
        n_seeds = group["Seed"].nunique()
        row = {
            "Config": config, "FeatureSet": feature_set, "ModelType": model_type,
            "Kind": kind, "Attack": attack, "N_Seeds": n_seeds,
        }
        for metric in metrics:
            vals = group[metric].dropna().to_numpy(dtype=float)
            key = metric.replace(" (%)", "").replace(" ", "_")
            row[f"{key}_Mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{key}_Std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["Config"] = pd.Categorical(summary["Config"], categories=MODEL_ORDER, ordered=True)
    return summary.sort_values(["Config", "FeatureSet", "Kind", "Attack"]).reset_index(drop=True)


def markdown_zero_day_table(summary: pd.DataFrame, feature_set: str) -> str:
    sub = summary[(summary["FeatureSet"] == feature_set) & (summary["Kind"] == "zero_day")]
    lines = [
        f"### Zero-day per-family Detection Rate / Precision / F1 (mean +/- std, feature_set={feature_set})",
        "",
        "| Config | " + " | ".join(ZERO_DAY_FAMILY_ORDER) + " |",
        "|---|" + "---|" * len(ZERO_DAY_FAMILY_ORDER),
    ]
    for config in [c for c in MODEL_ORDER if c in sub["Config"].astype(str).unique()]:
        cells = []
        for family in ZERO_DAY_FAMILY_ORDER:
            row = sub[(sub["Config"].astype(str) == config) & (sub["Attack"] == family)]
            if row.empty:
                cells.append("n/a")
                continue
            r = row.iloc[0]
            cells.append(
                f"DR {r['Detection_Rate_Mean']:.1f}+/-{r['Detection_Rate_Std']:.1f} "
                f"/ P {r['Precision_Mean']:.1f} / F1 {r['F1_Mean']:.1f} (n={int(r['N_Seeds'])})"
            )
        lines.append(f"| {config} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def append_to_paper_log(summary: pd.DataFrame) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_seeds_seen = sorted(summary["N_Seeds"].unique().tolist())
    header = (
        f"\n\n## Per-family report — {stamp} "
        f"(multiseed sweep, seed coverage up to n={max(n_seeds_seen) if n_seeds_seen else 0})\n\n"
        "Each zero-day family is scored Precision/Recall/F1 against the SAME shared "
        "known-test benign pool (see `evaluate_family_breakdown` in "
        "`agent/evaluation/metrics.py`), so Precision is comparable across families "
        "and configs. Full detail (including per-known-attack-class breakdown) is in "
        "`experiments/MULTISEED/per_family_summary.csv` / `.json`; this section is a "
        "zero-day-family summary snapshot, dated so later runs don't silently overwrite it.\n"
    )
    body = markdown_zero_day_table(summary, "ports")
    body += "\n" + markdown_zero_day_table(summary, "noports")

    if not PAPER_LOG.exists():
        PAPER_LOG.write_text("# Paper Results Log\n", encoding="utf-8")
    with PAPER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(header + body)
    print(f"Appended dated section to {PAPER_LOG}")


def main() -> None:
    long_df = load_long()
    long_df.to_csv(OUT_LONG, index=False)
    print(f"Wrote {len(long_df)} rows -> {OUT_LONG}")

    summary = build_summary(long_df)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    summary.to_json(OUT_SUMMARY_JSON, orient="records", indent=2)
    print(f"Wrote summary -> {OUT_SUMMARY_CSV} / {OUT_SUMMARY_JSON}")

    append_to_paper_log(summary)


if __name__ == "__main__":
    main()
