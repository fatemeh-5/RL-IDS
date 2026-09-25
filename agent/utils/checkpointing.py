"""Save / load experiment runs for reboot-safe workflows."""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf

from agent.data.constants import DEFAULT_B8_RUN, DEFAULT_B10_RUN, PROJECT_ROOT

# Keys that newer Keras versions serialize but older ones reject.
_STRIP_CONFIG_KEYS = {
    "quantization_config",
    "input_axes",
    "output_axes",
}


def _strip_unsupported_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: _strip_unsupported_keys(value)
            for key, value in obj.items()
            if key not in _STRIP_CONFIG_KEYS
        }
    if isinstance(obj, list):
        return [_strip_unsupported_keys(item) for item in obj]
    return obj


def safe_load_keras_model(model_path: str | Path):
    """
    Load a .keras model, including freezes saved with a newer Keras than
    the current environment (strips unsupported config keys when needed).
    """
    model_path = Path(model_path)
    try:
        return tf.keras.models.load_model(model_path)
    except TypeError as exc:
        message = str(exc)
        if "quantization_config" not in message and "Unrecognized keyword" not in message:
            raise

    if model_path.suffix != ".keras":
        raise RuntimeError(
            f"Compatible reload only supports .keras files, got: {model_path}"
        ) from exc

    with zipfile.ZipFile(model_path, "r") as source_zip:
        config = json.loads(source_zip.read("config.json"))
        cleaned = _strip_unsupported_keys(config)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            fixed_path = tmp_dir / model_path.name
            with zipfile.ZipFile(fixed_path, "w") as dest_zip:
                for info in source_zip.infolist():
                    if info.filename == "config.json":
                        dest_zip.writestr(
                            info,
                            json.dumps(cleaned).encode("utf-8"),
                        )
                    else:
                        dest_zip.writestr(info, source_zip.read(info.filename))
            return tf.keras.models.load_model(fixed_path)


@dataclass
class FrozenRun:
    run_dir: Path
    experiment_id: str
    online_model: Any
    target_model: Any | None
    configuration: dict[str, Any]
    per_attack: pd.DataFrame | None
    metrics: pd.DataFrame | None


def _find_first(run_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def discover_model_paths(run_dir: Path) -> dict[str, Path | None]:
    run_dir = Path(run_dir)
    online = _find_first(
        run_dir,
        [
            "*online_model.keras",
            "model/*.keras",
            "*_model.keras",
            "baseline_model.keras",
        ],
    )
    # Prefer online over generic if both exist
    online_pref = _find_first(run_dir, ["*online_model.keras"])
    if online_pref is not None:
        online = online_pref

    target = _find_first(run_dir, ["*target_model.keras"])
    config = _find_first(
        run_dir,
        [
            "*configuration.json",
            "configuration.json",
            "run_configuration.json",
        ],
    )
    per_attack = _find_first(
        run_dir,
        [
            "*zero_day_per_attack*.csv",
            "results/zero_day_per_attack_metrics.csv",
        ],
    )
    metrics = _find_first(
        run_dir,
        [
            "*evaluation_metrics.csv",
            "*known_test_metrics.csv",
            "results/known_test_metrics.csv",
        ],
    )
    return {
        "online": online,
        "target": target,
        "config": config,
        "per_attack": per_attack,
        "metrics": metrics,
    }


def load_run(run_dir: str | Path, load_target: bool = True) -> FrozenRun:
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    paths = discover_model_paths(run_dir)
    if paths["online"] is None:
        raise FileNotFoundError(f"No Keras model found under {run_dir}")

    online = safe_load_keras_model(paths["online"])
    target = None
    if load_target and paths["target"] is not None:
        target = safe_load_keras_model(paths["target"])

    configuration: dict[str, Any] = {}
    if paths["config"] is not None:
        with paths["config"].open("r", encoding="utf-8") as handle:
            configuration = json.load(handle)

    per_attack = (
        pd.read_csv(paths["per_attack"]) if paths["per_attack"] is not None else None
    )
    metrics = pd.read_csv(paths["metrics"]) if paths["metrics"] is not None else None

    experiment_id = str(
        configuration.get("experiment")
        or configuration.get("experiment_id")
        or run_dir.name
    )

    return FrozenRun(
        run_dir=run_dir,
        experiment_id=experiment_id,
        online_model=online,
        target_model=target,
        configuration=configuration,
        per_attack=per_attack,
        metrics=metrics,
    )


def load_baseline_runs() -> dict[str, FrozenRun]:
    """Load default frozen B8 and B10 baselines if present."""

    runs: dict[str, FrozenRun] = {}
    for key, path in (("B8", DEFAULT_B8_RUN), ("B10", DEFAULT_B10_RUN)):
        if path.exists():
            runs[key] = load_run(path)
            print(f"Loaded {key} from {path}")
        else:
            print(f"Skipped {key}: missing {path}")
    return runs


def save_run(
    run_dir: str | Path,
    *,
    experiment_id: str,
    online_model,
    target_model=None,
    configuration: dict[str, Any] | None = None,
    history_df: pd.DataFrame | None = None,
    eval_result: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    online_model.save(run_dir / f"{experiment_id.lower()}_online_model.keras")
    if target_model is not None:
        target_model.save(run_dir / f"{experiment_id.lower()}_target_model.keras")

    payload = {
        "experiment_id": experiment_id,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        **(configuration or {}),
    }
    with (run_dir / "configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if history_df is not None:
        history_df.to_csv(run_dir / "training_history.csv", index=False)

    if eval_result is not None:
        eval_result["metrics_df"].to_csv(run_dir / "evaluation_metrics.csv", index=False)
        eval_result["per_attack_df"].to_csv(
            run_dir / "zero_day_per_attack.csv", index=False
        )
        pd.DataFrame([eval_result["zero_day_metrics"]]).to_csv(
            run_dir / "zero_day_summary.csv", index=False
        )
        np_savez = run_dir / "predictions_and_q_values.npz"
        import numpy as np

        np.savez_compressed(
            np_savez,
            known_test_q_values=eval_result["known_test_q_values"],
            known_test_predictions=eval_result["known_test_predictions"],
            zero_day_q_values=eval_result["zero_day_q_values"],
            zero_day_predictions=eval_result["zero_day_predictions"],
        )

    print(f"Saved run to {run_dir}")
    return run_dir


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    episode: int,
    online_model,
    target_model=None,
    history_rows: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    online_model.save(checkpoint_dir / "online_model.keras")
    if target_model is not None:
        target_model.save(checkpoint_dir / "target_model.keras")

    state = {
        "episode": int(episode),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        **(extra or {}),
    }
    with (checkpoint_dir / "checkpoint.json").open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)

    if history_rows:
        pd.DataFrame(history_rows).to_csv(
            checkpoint_dir / "training_history.csv", index=False
        )

    return checkpoint_dir


def load_training_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    state_path = checkpoint_dir / "checkpoint.json"
    online_path = checkpoint_dir / "online_model.keras"
    if not state_path.exists() or not online_path.exists():
        return None

    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    online = safe_load_keras_model(online_path)
    target = None
    target_path = checkpoint_dir / "target_model.keras"
    if target_path.exists():
        target = safe_load_keras_model(target_path)

    history_path = checkpoint_dir / "training_history.csv"
    history_rows = []
    if history_path.exists():
        history_rows = pd.read_csv(history_path).to_dict(orient="records")

    return {
        "episode": int(state.get("episode", -1)),
        "online_model": online,
        "target_model": target,
        "history_rows": history_rows,
        "state": state,
        "checkpoint_dir": checkpoint_dir,
    }
