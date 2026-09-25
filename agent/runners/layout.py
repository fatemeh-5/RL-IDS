"""On-disk layout helpers that mirror Untitled freeze folders."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from agent.data.pipeline import PreparedData
from agent.runners.catalog import ExperimentSpec


def make_run_id(experiment_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{experiment_id}_{stamp}"


def create_run_dirs(
    spec: ExperimentSpec, run_id: str | None = None, run_dir: Path | None = None
) -> dict[str, Path | str]:
    run_id = run_id or make_run_id(spec.experiment_id)
    run_dir = run_dir or (spec.family_dir / run_id)
    model_dir = run_dir / "model"
    results_dir = run_dir / "results"
    artifacts_dir = run_dir / "artifacts"
    checkpoints_dir = run_dir / "checkpoints"
    for path in (model_dir, results_dir, artifacts_dir, checkpoints_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "model_dir": model_dir,
        "results_dir": results_dir,
        "artifacts_dir": artifacts_dir,
        "checkpoints_dir": checkpoints_dir,
    }


def save_standard_run(
    *,
    spec: ExperimentSpec,
    dirs: dict[str, Path],
    configuration: dict[str, Any],
    online_model,
    target_model=None,
    history_df: pd.DataFrame | None = None,
    eval_result: dict[str, Any] | None = None,
    data: PreparedData | None = None,
    model_filename: str = "online_model.keras",
    publish_top_level_csv: bool = True,
    mirror_family: bool = True,
) -> Path:
    """
    Write the same core artifacts Untitled produced:
      model/*.keras
      results/training_history.csv
      results/known_test_metrics.csv
      results/zero_day_metrics.csv
      results/zero_day_per_attack_metrics.csv
      results/*.npy
      artifacts/standard_scaler.joblib
      artifacts/feature_columns.json
      run_configuration.json
    """
    run_dir = dirs["run_dir"]
    model_dir = dirs["model_dir"]
    results_dir = dirs["results_dir"]
    artifacts_dir = dirs["artifacts_dir"]

    online_model.save(model_dir / model_filename)
    if target_model is not None:
        target_model.save(model_dir / "target_model.keras")

    with (run_dir / "run_configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2)

    if history_df is not None:
        history_df.to_csv(results_dir / "training_history.csv", index=False)

    if eval_result is not None:
        known = pd.DataFrame([eval_result["known_test_metrics"]])
        known.to_csv(results_dir / "known_test_metrics.csv", index=False)

        zero = pd.DataFrame([eval_result["zero_day_metrics"]])
        zero.to_csv(results_dir / "zero_day_metrics.csv", index=False)

        per_attack = eval_result["per_attack_df"].copy()
        # Untitled B6 compatibility columns
        if "Detection Rate (%)" in per_attack.columns:
            slim = per_attack[
                ["Attack", "Samples", "Detected", "Missed", "Detection Rate (%)", "Miss Rate (%)"]
            ]
        else:
            slim = per_attack
        slim.to_csv(results_dir / "zero_day_per_attack_metrics.csv", index=False)

        if "family_breakdown_df" in eval_result:
            eval_result["family_breakdown_df"].to_csv(
                results_dir / "family_breakdown_precision_recall_f1.csv", index=False
            )

        np.save(
            results_dir / "known_test_predictions.npy",
            eval_result["known_test_predictions"],
        )
        np.save(
            results_dir / "known_test_probabilities.npy",
            eval_result["known_test_q_values"].astype(np.float32),
        )
        np.save(
            results_dir / "zero_day_predictions.npy",
            eval_result["zero_day_predictions"],
        )
        np.save(
            results_dir / "zero_day_probabilities.npy",
            eval_result["zero_day_q_values"].astype(np.float32),
        )

        # Compatibility top-level CSV (B6 style) — skip for smoke overrides
        # and for multi-seed sweep runs (mirror_family=False), which must not
        # clobber the canonical single-run artifacts other tooling reads.
        if publish_top_level_csv and mirror_family and spec.compatibility_csv is not None:
            slim.to_csv(spec.compatibility_csv, index=False)

        # Family mirror: keep only the single canonical report CSV per model.
        # Top-level publish (compatibility_csv) is the one clean report.
        if mirror_family:
            family_results = spec.family_dir / "results"
            family_model = spec.family_dir / "model"
            family_artifacts = spec.family_dir / "artifacts"
            for path in (family_results, family_model, family_artifacts):
                path.mkdir(parents=True, exist_ok=True)
            slim.to_csv(family_results / "zero_day_per_attack_metrics.csv", index=False)
            if "family_breakdown_df" in eval_result:
                eval_result["family_breakdown_df"].to_csv(
                    family_results / "family_breakdown_precision_recall_f1.csv",
                    index=False,
                )
            shutil.copy2(model_dir / model_filename, family_model / model_filename)

    if data is not None:
        joblib.dump(data.scaler, artifacts_dir / "standard_scaler.joblib")
        with (artifacts_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
            json.dump(data.feature_columns, handle, indent=2)
        if mirror_family:
            family_artifacts = spec.family_dir / "artifacts"
            family_artifacts.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                artifacts_dir / "standard_scaler.joblib",
                family_artifacts / "standard_scaler.joblib",
            )
            shutil.copy2(
                artifacts_dir / "feature_columns.json",
                family_artifacts / "feature_columns.json",
            )

    print(f"Saved {spec.experiment_id} run -> {run_dir}")
    return run_dir
