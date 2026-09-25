#!/usr/bin/env python3
"""
Task 3 - Supervised baselines for the DRL-IDS paper.

Trains Random Forest, XGBoost, an MLP, and a stacked LSTM on the SAME
leak-free splits, the SAME train-only KMeans-SMOTE pool, and scores them
through the SAME `evaluate_all` used for the DRL agent. This guarantees the
"supervised vs DRL" comparison is apples-to-apples (no metric or split drift).

Design choices (so a reviewer can't dismiss the comparison):
  * Baselines train on `X_train_balanced` (train-only KMeans-SMOTE) by default,
    i.e. the exact augmented pool the agent draws episodes from. Use
    --train-set raw to instead train on the raw imbalanced train split
    (with class weighting) if you want to report both.
  * The MLP mirrors the agent's Q-network family -> isolates paradigm
    (supervised vs RL) with architecture held fixed.
  * The stacked LSTM mirrors the base paper's architecture as a plain classifier.

Place this file in Codes/scripts/ and run from the Codes/ directory, e.g.:
    python scripts/supervised_baselines.py                 # all 4, balanced pool
    python scripts/supervised_baselines.py --models rf xgb # just the tree models
    python scripts/supervised_baselines.py --train-set raw # raw split + class weights

Outputs land in experiments/SUPERVISED_BASELINES_<timestamp>/results/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --- make the `agent` package importable when run as scripts/... (mirrors prepare_data.py)
ROOT = Path(__file__).resolve().parents[1]  # Codes/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.data.constants import (
    EXPERIMENTS_DIR,
    RANDOM_STATE,
    ZERO_DAY_ATTACKS,
    feature_set_cache_dir,
    feature_set_columns,
)
from agent.data.pipeline import load_or_build_cache
from agent.evaluation.metrics import evaluate_all

ALL_MODELS = ["rf", "xgb", "mlp", "lstm"]


# --------------------------------------------------------------------------- #
# Adapters: make every classifier speak the metric code's .predict() contract  #
# --------------------------------------------------------------------------- #
class ProbaAdapter:
    """Wrap a sklearn/xgboost classifier as a Keras-style predictor.

    Returns an (N, 2) float array; column 1 is P(attack), matching how
    evaluate_binary_split reads q_values[:, 1] for ROC-AUC.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X, batch_size: int = 8192, verbose: int = 0) -> np.ndarray:
        proba = self.model.predict_proba(np.asarray(X, dtype=np.float32))
        return np.asarray(proba, dtype=np.float32)


class KerasSeqAdapter:
    """Wrap a sequence model so evaluate_all can feed it flat (N, F) arrays.

    Reshapes (N, F) -> (N, F, 1) inside predict so the LSTM sees a length-F
    sequence of 1 feature each.
    """

    def __init__(self, model, n_features: int):
        self.model = model
        self.n_features = n_features

    def predict(self, X, batch_size: int = 8192, verbose: int = 0) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32).reshape(-1, self.n_features, 1)
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)


# --------------------------------------------------------------------------- #
# Model builders                                                               #
# --------------------------------------------------------------------------- #
def build_random_forest(balanced: bool, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=25,            # bounded so trees don't blow up on millions of rows
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
        # SMOTE already balances the pool; only weight when training on raw split
        class_weight=None if balanced else "balanced_subsample",
    )


def build_xgboost(balanced: bool, seed: int, scale_pos_weight: float):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        eval_metric="logloss",
        scale_pos_weight=1.0 if balanced else scale_pos_weight,
    )


def build_mlp(input_dim: int, hidden: list[int], seed: int):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(name="Supervised_MLP")
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for units in hidden:
        model.add(tf.keras.layers.Dense(units, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(input_dim: int, units: int, seed: int):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(name="Supervised_StackedLSTM")
    model.add(tf.keras.layers.Input(shape=(input_dim, 1)))
    model.add(tf.keras.layers.LSTM(units, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------- #
# Training helpers                                                             #
# --------------------------------------------------------------------------- #
def train_keras(model, X_tr, y_tr, X_val, y_val, *, seq, n_features,
                class_weight, epochs, batch_size):
    import tensorflow as tf

    if seq:
        X_tr = np.asarray(X_tr, dtype=np.float32).reshape(-1, n_features, 1)
        X_val = np.asarray(X_val, dtype=np.float32).reshape(-1, n_features, 1)

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early],
        verbose=2,
    )
    return model


def summarize(name: str, result: dict) -> tuple[dict, pd.DataFrame]:
    """Pull the headline row + per-family table from an evaluate_all result.

    Weighted Zero-Day = overall detection rate (support-weighted across families).
    Macro Zero-Day    = unweighted mean of per-family detection rates.
    Same definitions used for the B0-B10 DRL runs.
    """
    kt = result["known_test_metrics"]
    zd = result["zero_day_metrics"]
    per = result["per_attack_df"]

    macro_zd = float(per["Detection Rate"].mean() * 100.0)
    weighted_zd = float(zd["Detection_Rate"] * 100.0)

    row = {
        "Model": name,
        "KT_Accuracy": round(kt["Accuracy"] * 100, 2),
        "KT_Precision": round(kt["Precision_Attack"] * 100, 2),
        "KT_Recall": round(kt["Recall_Attack"] * 100, 2),
        "KT_F1": round(kt["F1_Attack"] * 100, 2),
        "KT_ROC_AUC": round(kt["ROC_AUC"] * 100, 2),
        "Weighted_ZeroDay_%": round(weighted_zd, 2),
        "Macro_ZeroDay_%": round(macro_zd, 2),
    }
    per = per.copy()
    per.insert(0, "Model", name)
    return row, per


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised baselines for DRL-IDS.")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS, help="Which baselines to run.")
    parser.add_argument("--train-set", choices=["balanced", "raw"], default="balanced",
                        help="balanced = train-only KMeans-SMOTE pool (default); "
                             "raw = raw imbalanced train split + class weighting.")
    parser.add_argument("--tree-subsample", type=int, default=None,
                        help="Optional row cap for RF/XGB training only "
                             "(e.g. 2000000) if the tree models are too slow. "
                             "Keras models always use the full pool.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--mlp-hidden", type=int, nargs="+", default=[128, 64],
                        help="Set these to match your DQN's hidden layers for the "
                             "cleanest supervised-vs-RL comparison.")
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--feature-set", choices=["ports", "noports"], default="ports",
                        help="ports = original 10 columns (default); "
                             "noports = drops L4_SRC_PORT/L4_DST_PORT.")
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="Write results directly here instead of creating a "
                              "new experiments/SUPERVISED_BASELINES_<ts>/results dir "
                              "(used by the multi-seed sweep).")
    args = parser.parse_args()

    balanced = args.train_set == "balanced"
    rng = np.random.default_rng(args.seed)

    print(f"Loading prepared cache (feature_set={args.feature_set}, require_balanced={balanced}) ...")
    data = load_or_build_cache(
        cache_dir=feature_set_cache_dir(args.feature_set),
        require_balanced=balanced,
        feature_columns=feature_set_columns(args.feature_set),
    )
    n_features = data.X_train_scaled.shape[1]

    # choose the training pool
    if balanced:
        X_full = np.asarray(data.X_train_balanced, dtype=np.float32)
        y_full = np.asarray(data.y_train_balanced, dtype=np.int8)
    else:
        X_full = np.asarray(data.X_train_scaled, dtype=np.float32)
        y_full = np.asarray(data.y_train, dtype=np.int8)

    n_pos = int((y_full == 1).sum())
    n_neg = int((y_full == 0).sum())
    spw = (n_neg / max(n_pos, 1))
    print(f"Training pool: {len(y_full):,} rows "
          f"(attack={n_pos:,}, benign={n_neg:,}, scale_pos_weight={spw:.3f})")

    # class weights for keras when training on raw
    keras_class_weight = None
    if not balanced:
        from sklearn.utils.class_weight import compute_class_weight
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_full)
        keras_class_weight = {0: float(w[0]), 1: float(w[1])}

    # optional subsample for the tree models only
    if args.tree_subsample and args.tree_subsample < len(y_full):
        idx = rng.choice(len(y_full), size=args.tree_subsample, replace=False)
        X_tree, y_tree = X_full[idx], y_full[idx]
        print(f"Tree models use a subsample of {len(y_tree):,} rows.")
    else:
        X_tree, y_tree = X_full, y_full

    summary_rows: list[dict] = []
    per_family_tables: list[pd.DataFrame] = []
    timings: dict[str, float] = {}

    for name in args.models:
        print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")
        t0 = time.time()
        try:
            if name == "rf":
                clf = build_random_forest(balanced, args.seed)
                clf.fit(X_tree, y_tree)
                model = ProbaAdapter(clf)

            elif name == "xgb":
                clf = build_xgboost(balanced, args.seed, spw)
                clf.fit(X_tree, y_tree)
                model = ProbaAdapter(clf)

            elif name == "mlp":
                net = build_mlp(n_features, args.mlp_hidden, args.seed)
                net = train_keras(net, X_full, y_full,
                                  data.X_valid_scaled, data.y_valid,
                                  seq=False, n_features=n_features,
                                  class_weight=keras_class_weight,
                                  epochs=args.epochs, batch_size=args.batch_size)
                model = net  # already speaks .predict(X, batch_size, verbose)

            elif name == "lstm":
                net = build_lstm(n_features, args.lstm_units, args.seed)
                net = train_keras(net, X_full, y_full,
                                  data.X_valid_scaled, data.y_valid,
                                  seq=True, n_features=n_features,
                                  class_weight=keras_class_weight,
                                  epochs=args.epochs, batch_size=args.batch_size)
                model = KerasSeqAdapter(net, n_features)
            else:
                continue

            result = evaluate_all(model, data, batch_size=args.batch_size, verbose=0)
            row, per = summarize(name.upper(), result)
            summary_rows.append(row)
            per_family_tables.append(per)
            timings[name] = time.time() - t0

            print(f"[{name}] Known-Test Recall={row['KT_Recall']}%  "
                  f"ROC-AUC={row['KT_ROC_AUC']}%  |  "
                  f"Weighted-ZD={row['Weighted_ZeroDay_%']}%  "
                  f"Macro-ZD={row['Macro_ZeroDay_%']}%  "
                  f"({timings[name]:.0f}s)")

        except ImportError as exc:
            print(f"[skip {name}] missing dependency: {exc}")
        except Exception as exc:  # noqa: BLE001 - keep the run going, report the rest
            print(f"[error {name}] {type(exc).__name__}: {exc}")

    if not summary_rows:
        print("\nNo models produced results.")
        return

    summary_df = pd.DataFrame(summary_rows)
    per_family_df = pd.concat(per_family_tables, ignore_index=True)

    print("\n" + "=" * 60)
    print("SUPERVISED BASELINE SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # persist, mirroring the experiments/<...>/results/ layout (or --out-dir
    # verbatim when the multi-seed sweep drives this script per-model/cell)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(EXPERIMENTS_DIR) / f"SUPERVISED_BASELINES_{ts}" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    per_family_df.to_csv(out_dir / "per_family_detection.csv", index=False)
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "train_set": args.train_set,
            "seed": args.seed,
            "feature_set": args.feature_set,
            "models": args.models,
            "n_features": n_features,
            "mlp_hidden": args.mlp_hidden,
            "lstm_units": args.lstm_units,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "tree_subsample": args.tree_subsample,
            "pool_size": int(len(y_full)),
            "pool_attack": n_pos,
            "pool_benign": n_neg,
            "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
        }, handle, indent=2)

    print(f"\nSaved -> {out_dir}")

    # Sweep marker: one flat METRICS_ROW.json per (model, feature_set, seed),
    # same schema as the DRL side (agent/../run_drl_cell.py), so
    # aggregate_multiseed.py can scan both without caring which trainer ran.
    if args.out_dir is not None and len(summary_rows) == 1:
        row_raw = summary_rows[0]
        per_indexed = per_family_tables[0].set_index("Attack")["Detection Rate (%)"]
        family_rates = {
            attack: float(per_indexed.get(attack, float("nan")))
            for attack in ZERO_DAY_ATTACKS
        }
        row = {
            "config": row_raw["Model"],
            "feature_set": args.feature_set,
            "model_type": "supervised",
            "seed": args.seed,
            "KT_Accuracy": row_raw["KT_Accuracy"],
            "KT_Precision": row_raw["KT_Precision"],
            "KT_Recall": row_raw["KT_Recall"],
            "KT_F1": row_raw["KT_F1"],
            "KT_ROC_AUC": row_raw["KT_ROC_AUC"],
            "Weighted_ZeroDay": row_raw["Weighted_ZeroDay_%"],
            "Macro_ZeroDay": row_raw["Macro_ZeroDay_%"],
            **family_rates,
        }
        (out_dir / "METRICS_ROW.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"Wrote {out_dir / 'METRICS_ROW.json'}")


if __name__ == "__main__":
    main()