#!/usr/bin/env python3
"""Run ONE multi-seed sweep cell for a DRL baseline (B0-B10).

Trains one config at one feature_set/seed, evaluates it, and writes a flat
METRICS_ROW.json into --out-dir (consumed later by aggregate_multiseed.py).
Designed to be launched as an isolated subprocess per cell so a crash in one
cell (OOM, NaN loss, etc.) can't take down the rest of a multi-day sweep, and
so re-running is just "does METRICS_ROW.json already exist".

Usage:
    python scripts/run_drl_cell.py --config B10 --feature-set noports \
        --seed 43 --out-dir experiments/MULTISEED/runs/drl/B10/noports/seed43
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

# Must be set before TensorFlow is imported (transitively, via agent.runners.runners
# -> agent.training.*) to take effect: forces deterministic GPU op implementations
# (cuDNN LSTM etc.) so a fixed seed reproduces bit-identical runs on GPU, not just CPU.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.data.constants import ZERO_DAY_ATTACKS
from agent.runners.runners import rebuild_experiment
from agent.utils import progress


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one DRL multi-seed sweep cell.")
    parser.add_argument("--config", required=True, help="B0..B11")
    parser.add_argument("--feature-set", required=True, choices=["ports", "noports"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=None,
                         help="Override config episodes (smoke-testing only).")
    args = parser.parse_args()
    random.seed(args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "METRICS_ROW.json"
    if marker.exists():
        print(f"Already done: {marker}")
        return 0

    tee = progress.install(progress.cell_label(args.config, args.feature_set, args.seed))
    tee.event("START training")
    try:
        out = rebuild_experiment(
            args.config,
            feature_set=args.feature_set,
            seed_override=args.seed,
            episodes_override=args.episodes,
            run_dir=out_dir,
            mirror_family=False,
            publish_top_level_csv=False,
        )
    except Exception:  # noqa: BLE001 - surface full traceback, exit nonzero
        traceback.print_exc()
        tee.event(f"FAILED after {tee.elapsed()} -- see {out_dir / 'log.txt'}")
        (out_dir / "FAILED.json").write_text(
            json.dumps({"error": traceback.format_exc()}, indent=2), encoding="utf-8"
        )
        return 1

    eval_result = out["eval_result"]
    kt = eval_result["known_test_metrics"]
    zd = eval_result["zero_day_metrics"]
    per = eval_result["per_attack_df"]
    fam = eval_result["family_breakdown_df"]

    macro_zd = float(per["Detection Rate (%)"].mean())
    weighted_zd = float(zd["Detection_Rate"]) * 100.0

    family_rates = {}
    fam_zd = fam[fam["Kind"] == "zero_day"].set_index("Attack")
    for attack in ZERO_DAY_ATTACKS:
        if attack in fam_zd.index:
            family_rates[attack] = float(fam_zd.loc[attack, "Detection Rate (%)"])
            family_rates[f"{attack}_Precision"] = float(fam_zd.loc[attack, "Precision (%)"])
            family_rates[f"{attack}_F1"] = float(fam_zd.loc[attack, "F1 (%)"])
        else:
            family_rates[attack] = float("nan")
            family_rates[f"{attack}_Precision"] = float("nan")
            family_rates[f"{attack}_F1"] = float("nan")

    row = {
        "config": args.config,
        "feature_set": args.feature_set,
        "model_type": "drl",
        "seed": args.seed,
        "KT_Accuracy": float(kt["Accuracy"]) * 100.0,
        "KT_Precision": float(kt["Precision_Attack"]) * 100.0,
        "KT_Recall": float(kt["Recall_Attack"]) * 100.0,
        "KT_F1": float(kt["F1_Attack"]) * 100.0,
        "KT_ROC_AUC": float(kt["ROC_AUC"]) * 100.0,
        "Weighted_ZeroDay": weighted_zd,
        "Macro_ZeroDay": macro_zd,
        **family_rates,
    }
    marker.write_text(json.dumps(row, indent=2), encoding="utf-8")
    tee.event(
        f"DONE in {tee.elapsed()} | KT_F1={row['KT_F1']:.2f} "
        f"KT_AUC={row['KT_ROC_AUC']:.2f} | ZD weighted={weighted_zd:.2f} macro={macro_zd:.2f}"
    )
    print(f"Wrote {marker}")
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
