"""Shared constants for paths, splits, features, and zero-day holdouts."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Codes/ (package is agent/data/)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PREPARED_DATA_DIR = DATA_DIR / "prepared"

DATASET_PATH = RAW_DATA_DIR / "NF-UQ-NIDS.csv"
CACHE_DIR = PREPARED_DATA_DIR

# Training outputs / one CSV report per model live here
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR
RESULTS_DIR = EXPERIMENTS_DIR
FROZEN_DIR = EXPERIMENTS_DIR / "frozen"

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

ZERO_DAY_ATTACKS = [
    "Shellcode",
    "Brute Force",
    "Theft",
    "ransomware",
    "Backdoor",
]

IDENTIFIER_COLUMNS = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
]

METADATA_COLUMNS = [
    "Attack",
    "Dataset",
]

TARGET_COLUMN = "Label"

BASELINE_FEATURE_COLUMNS = [
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
]

# Multi-seed ports-vs-noports ablation: "ports" is the original feature set
# above; "noports" drops the two port columns (a known leakage/overfitting
# concern in NIDS flow features). "ports" reuses the existing data/prepared/
# cache dir (unchanged, no re-prep needed); "noports" gets its own cache dir
# since it has a different column count / scaler fit.
PORTS_FEATURE_COLUMNS = list(BASELINE_FEATURE_COLUMNS)
NOPORTS_FEATURE_COLUMNS = [
    c for c in BASELINE_FEATURE_COLUMNS if c not in ("L4_SRC_PORT", "L4_DST_PORT")
]
FEATURE_SETS: dict[str, list[str]] = {
    "ports": PORTS_FEATURE_COLUMNS,
    "noports": NOPORTS_FEATURE_COLUMNS,
}
CACHE_DIR_BY_FEATURE_SET: dict[str, Path] = {
    "ports": PREPARED_DATA_DIR,  # existing flat cache dir, unchanged
    "noports": PREPARED_DATA_DIR / "noports",
}


def feature_set_columns(feature_set: str) -> list[str]:
    try:
        return FEATURE_SETS[feature_set]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature_set {feature_set!r}; expected one of {sorted(FEATURE_SETS)}"
        ) from exc


def feature_set_cache_dir(feature_set: str) -> Path:
    try:
        return CACHE_DIR_BY_FEATURE_SET[feature_set]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature_set {feature_set!r}; expected one of {sorted(FEATURE_SETS)}"
        ) from exc

KMEANS_CLUSTERS = 1000
KMEANS_BATCH_SIZE = 32768
KMEANS_NEIGHBORS = 5

DEFAULT_B8_RUN = FROZEN_DIR / "B8_hybrid_sampling" / "B8_20260806_000816"
DEFAULT_B10_RUN = FROZEN_DIR / "B10_restrained_adaptive" / "B10_20260806_012530"
