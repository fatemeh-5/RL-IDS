"""Prepare and cache train/valid/test/zero-day arrays for reboot-safe reuse."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from agent.data.constants import (
    BASELINE_FEATURE_COLUMNS,
    CACHE_DIR,
    DATASET_PATH,
    KMEANS_BATCH_SIZE,
    KMEANS_CLUSTERS,
    KMEANS_NEIGHBORS,
    RANDOM_STATE,
    TARGET_COLUMN,
    ZERO_DAY_ATTACKS,
)

META_FILENAME = "meta.json"
SCALER_FILENAME = "standard_scaler.joblib"
FEATURES_FILENAME = "feature_columns.json"


@dataclass
class PreparedData:
    X_train_scaled: np.ndarray
    y_train: np.ndarray
    train_attack: np.ndarray
    X_valid_scaled: np.ndarray
    y_valid: np.ndarray
    valid_attack: np.ndarray
    X_known_test_scaled: np.ndarray
    y_known_test: np.ndarray
    known_test_attack: np.ndarray
    X_zero_day_scaled: np.ndarray
    y_zero_day: np.ndarray
    zero_day_attack: np.ndarray
    X_train_balanced: np.ndarray | None
    y_train_balanced: np.ndarray | None
    feature_columns: list[str]
    scaler: StandardScaler
    meta: dict[str, Any]
    cache_dir: Path
    train_dataset: np.ndarray | None = None
    valid_dataset: np.ndarray | None = None
    known_test_dataset: np.ndarray | None = None
    zero_day_dataset: np.ndarray | None = None

    @property
    def has_balanced(self) -> bool:
        return self.X_train_balanced is not None and self.y_train_balanced is not None

    @property
    def has_dataset_column(self) -> bool:
        return self.train_dataset is not None


def cache_is_ready(cache_dir: Path | None = None, require_balanced: bool = True) -> bool:
    cache_dir = Path(cache_dir or CACHE_DIR)
    required = [
        cache_dir / META_FILENAME,
        cache_dir / SCALER_FILENAME,
        cache_dir / "X_train_scaled.npy",
        cache_dir / "y_train.npy",
        cache_dir / "X_valid_scaled.npy",
        cache_dir / "y_valid.npy",
        cache_dir / "X_known_test_scaled.npy",
        cache_dir / "y_known_test.npy",
        cache_dir / "X_zero_day_scaled.npy",
        cache_dir / "y_zero_day.npy",
        cache_dir / "zero_day_attack.npy",
    ]
    if require_balanced:
        required.extend(
            [
                cache_dir / "X_train_balanced.npy",
                cache_dir / "y_train_balanced.npy",
            ]
        )
    return all(path.exists() for path in required)


def _save_array(path: Path, array: np.ndarray) -> None:
    np.save(path, array)


def _working_columns(feature_columns: list[str]) -> list[str]:
    return list(feature_columns) + [TARGET_COLUMN, "Attack", "Dataset"]


def prepare_and_cache(
    dataset_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    random_state: int = RANDOM_STATE,
    max_rows: int | None = None,
    skip_smote: bool = False,
    force: bool = False,
    feature_columns: list[str] | None = None,
) -> PreparedData:
    """Build scaled splits (+ optional KMeans-SMOTE) and write cache/prepared."""

    dataset_path = Path(dataset_path or DATASET_PATH)
    cache_dir = Path(cache_dir or CACHE_DIR)
    feature_columns = list(feature_columns or BASELINE_FEATURE_COLUMNS)

    if not force and cache_is_ready(cache_dir, require_balanced=not skip_smote):
        print(f"Cache already ready at {cache_dir}")
        return load_prepared_cache(cache_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(f"Loading {dataset_path} ...")
    raw_df = pd.read_csv(dataset_path)
    if max_rows is not None and max_rows > 0:
        raw_df = raw_df.sample(n=min(max_rows, len(raw_df)), random_state=random_state)
        print(f"Using subsample of {len(raw_df):,} rows (max_rows={max_rows})")

    known_df = raw_df[~raw_df["Attack"].isin(ZERO_DAY_ATTACKS)].copy()
    zero_day_df = raw_df[raw_df["Attack"].isin(ZERO_DAY_ATTACKS)].copy()

    working = _working_columns(feature_columns)
    known_model_df = known_df[working].copy()
    zero_day_model_df = zero_day_df[working].copy()

    train_df, temp_df = train_test_split(
        known_model_df,
        test_size=0.30,
        random_state=random_state,
        stratify=known_model_df[TARGET_COLUMN],
    )
    valid_df, known_test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df[TARGET_COLUMN],
    )

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COLUMN]
    X_known_test = known_test_df[feature_columns]
    y_known_test = known_test_df[TARGET_COLUMN]
    X_zero_day = zero_day_model_df[feature_columns]
    y_zero_day = zero_day_model_df[TARGET_COLUMN]

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_valid_scaled = scaler.transform(X_valid).astype(np.float32)
    X_known_test_scaled = scaler.transform(X_known_test).astype(np.float32)
    X_zero_day_scaled = scaler.transform(X_zero_day).astype(np.float32)

    y_train_arr = y_train.to_numpy(dtype=np.int8)
    y_valid_arr = y_valid.to_numpy(dtype=np.int8)
    y_known_test_arr = y_known_test.to_numpy(dtype=np.int8)
    y_zero_day_arr = y_zero_day.to_numpy(dtype=np.int8)

    train_attack = train_df["Attack"].to_numpy(dtype=object)
    valid_attack = valid_df["Attack"].to_numpy(dtype=object)
    known_test_attack = known_test_df["Attack"].to_numpy(dtype=object)
    zero_day_attack = zero_day_model_df["Attack"].to_numpy(dtype=object)

    train_dataset = train_df["Dataset"].to_numpy(dtype=object)
    valid_dataset = valid_df["Dataset"].to_numpy(dtype=object)
    known_test_dataset = known_test_df["Dataset"].to_numpy(dtype=object)
    zero_day_dataset = zero_day_model_df["Dataset"].to_numpy(dtype=object)

    X_train_balanced = None
    y_train_balanced = None

    if not skip_smote:
        from imblearn.over_sampling import KMeansSMOTE
        from sklearn.cluster import MiniBatchKMeans

        print("Running memory-safe KMeans-SMOTE on training set ...")
        smote_started = time.time()
        kmeans = MiniBatchKMeans(
            n_clusters=KMEANS_CLUSTERS,
            batch_size=KMEANS_BATCH_SIZE,
            random_state=random_state,
            n_init=3,
            reassignment_ratio=0.01,
        )
        sampler = KMeansSMOTE(
            kmeans_estimator=kmeans,
            random_state=random_state,
            k_neighbors=KMEANS_NEIGHBORS,
            cluster_balance_threshold="auto",
            n_jobs=-1,
        )
        X_train_balanced, y_train_balanced = sampler.fit_resample(
            X_train_scaled,
            y_train_arr,
        )
        X_train_balanced = np.asarray(X_train_balanced, dtype=np.float32)
        y_train_balanced = np.asarray(y_train_balanced, dtype=np.int8)
        print(
            f"SMOTE done in {time.time() - smote_started:.1f}s "
            f"-> {len(X_train_balanced):,} balanced samples"
        )
    else:
        print("Skipping SMOTE (skip_smote=True); using scaled train as balanced.")
        X_train_balanced = X_train_scaled.copy()
        y_train_balanced = y_train_arr.copy()

    # Persist arrays
    _save_array(cache_dir / "X_train_scaled.npy", X_train_scaled)
    _save_array(cache_dir / "y_train.npy", y_train_arr)
    _save_array(cache_dir / "train_attack.npy", train_attack)
    _save_array(cache_dir / "X_valid_scaled.npy", X_valid_scaled)
    _save_array(cache_dir / "y_valid.npy", y_valid_arr)
    _save_array(cache_dir / "valid_attack.npy", valid_attack)
    _save_array(cache_dir / "X_known_test_scaled.npy", X_known_test_scaled)
    _save_array(cache_dir / "y_known_test.npy", y_known_test_arr)
    _save_array(cache_dir / "known_test_attack.npy", known_test_attack)
    _save_array(cache_dir / "X_zero_day_scaled.npy", X_zero_day_scaled)
    _save_array(cache_dir / "y_zero_day.npy", y_zero_day_arr)
    _save_array(cache_dir / "zero_day_attack.npy", zero_day_attack)

    _save_array(cache_dir / "train_dataset.npy", train_dataset)
    _save_array(cache_dir / "valid_dataset.npy", valid_dataset)
    _save_array(cache_dir / "known_test_dataset.npy", known_test_dataset)
    _save_array(cache_dir / "zero_day_dataset.npy", zero_day_dataset)

    if X_train_balanced is not None:
        _save_array(cache_dir / "X_train_balanced.npy", X_train_balanced)
        _save_array(cache_dir / "y_train_balanced.npy", y_train_balanced)

    joblib.dump(scaler, cache_dir / SCALER_FILENAME)
    with (cache_dir / FEATURES_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(feature_columns, handle, indent=2)

    meta = {
        "dataset_path": str(dataset_path.resolve()),
        "random_state": random_state,
        "max_rows": max_rows,
        "skip_smote": skip_smote,
        "zero_day_attacks": ZERO_DAY_ATTACKS,
        "feature_columns": feature_columns,
        "n_train": int(len(X_train_scaled)),
        "n_valid": int(len(X_valid_scaled)),
        "n_known_test": int(len(X_known_test_scaled)),
        "n_zero_day": int(len(X_zero_day_scaled)),
        "n_train_balanced": (
            None if X_train_balanced is None else int(len(X_train_balanced))
        ),
        "source_datasets": sorted(str(d) for d in pd.unique(raw_df["Dataset"])),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with (cache_dir / META_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"Cached prepared data under {cache_dir} in {meta['elapsed_seconds']}s")
    return load_prepared_cache(cache_dir)


def load_prepared_cache(cache_dir: Path | str | None = None) -> PreparedData:
    cache_dir = Path(cache_dir or CACHE_DIR)
    if not (cache_dir / META_FILENAME).exists():
        raise FileNotFoundError(
            f"No prepared cache at {cache_dir}. Run: python scripts/prepare_data.py"
        )

    with (cache_dir / META_FILENAME).open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    scaler = joblib.load(cache_dir / SCALER_FILENAME)
    feature_columns = list(meta.get("feature_columns", BASELINE_FEATURE_COLUMNS))

    balanced_x = cache_dir / "X_train_balanced.npy"
    balanced_y = cache_dir / "y_train_balanced.npy"
    X_train_balanced = np.load(balanced_x) if balanced_x.exists() else None
    y_train_balanced = np.load(balanced_y) if balanced_y.exists() else None

    def _load_optional(name: str) -> np.ndarray | None:
        path = cache_dir / name
        return np.load(path, allow_pickle=True) if path.exists() else None

    return PreparedData(
        X_train_scaled=np.load(cache_dir / "X_train_scaled.npy"),
        y_train=np.load(cache_dir / "y_train.npy"),
        train_attack=np.load(cache_dir / "train_attack.npy", allow_pickle=True),
        X_valid_scaled=np.load(cache_dir / "X_valid_scaled.npy"),
        y_valid=np.load(cache_dir / "y_valid.npy"),
        valid_attack=np.load(cache_dir / "valid_attack.npy", allow_pickle=True),
        X_known_test_scaled=np.load(cache_dir / "X_known_test_scaled.npy"),
        y_known_test=np.load(cache_dir / "y_known_test.npy"),
        known_test_attack=np.load(cache_dir / "known_test_attack.npy", allow_pickle=True),
        X_zero_day_scaled=np.load(cache_dir / "X_zero_day_scaled.npy"),
        y_zero_day=np.load(cache_dir / "y_zero_day.npy"),
        zero_day_attack=np.load(cache_dir / "zero_day_attack.npy", allow_pickle=True),
        X_train_balanced=X_train_balanced,
        y_train_balanced=y_train_balanced,
        feature_columns=feature_columns,
        scaler=scaler,
        meta=meta,
        cache_dir=cache_dir,
        train_dataset=_load_optional("train_dataset.npy"),
        valid_dataset=_load_optional("valid_dataset.npy"),
        known_test_dataset=_load_optional("known_test_dataset.npy"),
        zero_day_dataset=_load_optional("zero_day_dataset.npy"),
    )


def load_or_build_cache(
    cache_dir: Path | str | None = None,
    require_balanced: bool = True,
    **prepare_kwargs: Any,
) -> PreparedData:
    cache_dir = Path(cache_dir or CACHE_DIR)
    if cache_is_ready(cache_dir, require_balanced=require_balanced):
        print(f"Loading cache from {cache_dir}")
        return load_prepared_cache(cache_dir)
    print("Cache missing — building now ...")
    return prepare_and_cache(cache_dir=cache_dir, skip_smote=not require_balanced, **prepare_kwargs)
