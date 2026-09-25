"""Evaluation helpers for known-test and zero-day splits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from agent.data.pipeline import PreparedData


def _match_model_input_width(model, X_arr: np.ndarray) -> np.ndarray:
    """Pad flow features with neutral meta-features when the model expects
    more input columns than the raw flow data provides.

    Models trained on an augmented-state env (e.g. B11's InvestigateIDSEnvironment,
    which appends [budget_remaining_norm, post_alert_countdown_norm] to the 10
    flow features) can't be fed raw flow features directly at eval time — there
    is no episode/investigate loop during static known-test/zero-day evaluation.
    We pad with the neutral state: full investigate budget available (1.0) and
    no active post-alert countdown (0.0), so evaluation reflects a fresh,
    unbiased decision rather than leaking any mid-episode meta-state.
    """
    expected = int(model.input_shape[-1])
    actual = X_arr.shape[-1]
    if expected == actual:
        return X_arr
    if expected < actual:
        raise ValueError(
            f"Model expects {expected} input features but got {actual}; cannot truncate."
        )
    pad = np.zeros((X_arr.shape[0], expected - actual), dtype=X_arr.dtype)
    if pad.shape[1] >= 1:
        pad[:, 0] = 1.0  # budget_remaining_norm = full budget
    return np.concatenate([X_arr, pad], axis=1)


def evaluate_binary_split(
    model,
    X,
    y,
    split_name: str,
    batch_size: int = 8192,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int8)
    X_arr = _match_model_input_width(model, X_arr)

    q_values = model.predict(X_arr, batch_size=batch_size, verbose=verbose)
    # A 3-action model (B11: benign/attack/investigate) must never surface
    # "investigate" as a final classification at inference time — restrict
    # argmax to {benign, attack}. ROC-AUC still uses q_values[:, 1] as the
    # attack score, which is unaffected by the extra column.
    if q_values.shape[-1] == 3:
        predictions = np.argmax(q_values[:, :2], axis=1).astype(np.int8)
    else:
        predictions = np.argmax(q_values, axis=1).astype(np.int8)

    cm = confusion_matrix(y_arr, predictions, labels=[0, 1])
    metrics = {
        "Split": split_name,
        "Accuracy": float(accuracy_score(y_arr, predictions)),
        "Precision_Attack": float(precision_score(y_arr, predictions, zero_division=0)),
        "Recall_Attack": float(recall_score(y_arr, predictions, zero_division=0)),
        "F1_Attack": float(f1_score(y_arr, predictions, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_arr, q_values[:, 1])),
        "True_Negative": int(cm[0, 0]),
        "False_Positive": int(cm[0, 1]),
        "False_Negative": int(cm[1, 0]),
        "True_Positive": int(cm[1, 1]),
    }
    return q_values, predictions, metrics, cm


def evaluate_zero_day_overall(
    model,
    X,
    batch_size: int = 8192,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    X_arr = np.asarray(X, dtype=np.float32)
    X_arr = _match_model_input_width(model, X_arr)
    q_values = model.predict(X_arr, batch_size=batch_size, verbose=verbose)
    if q_values.shape[-1] == 3:
        predictions = np.argmax(q_values[:, :2], axis=1).astype(np.int8)
    else:
        predictions = np.argmax(q_values, axis=1).astype(np.int8)

    detected = int((predictions == 1).sum())
    missed = int((predictions == 0).sum())
    total = len(predictions)
    metrics = {
        "Split": "Zero-Day",
        "Samples": total,
        "Detected": detected,
        "Missed": missed,
        "Detection_Rate": detected / total if total else 0.0,
        "Miss_Rate": missed / total if total else 0.0,
    }
    return q_values, predictions, metrics


def evaluate_per_attack(
    attack_names,
    y_true,
    y_pred,
) -> pd.DataFrame:
    attack_names = np.asarray(attack_names)
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    if not (len(attack_names) == len(y_true) == len(y_pred)):
        raise ValueError("Attack names, labels, and predictions must align.")

    rows = []
    for attack in sorted(np.unique(attack_names)):
        mask = attack_names == attack
        samples = int(mask.sum())
        detected = int(np.sum(y_pred[mask] == 1))
        missed = int(np.sum(y_pred[mask] == 0))
        detection_rate = detected / samples if samples else 0.0
        rows.append(
            {
                "Attack": attack,
                "Samples": samples,
                "Detected": detected,
                "Missed": missed,
                "Detection Rate": detection_rate,
                "Miss Rate": 1.0 - detection_rate,
                "Detection Rate (%)": detection_rate * 100,
                "Miss Rate (%)": (1.0 - detection_rate) * 100,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("Samples", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_all(
    model,
    data: PreparedData,
    batch_size: int = 8192,
    verbose: int = 0,
) -> dict[str, Any]:
    """Run validation, known-test, and zero-day (+ per-attack) evaluation."""

    _, _, valid_metrics, valid_cm = evaluate_binary_split(
        model, data.X_valid_scaled, data.y_valid, "Validation", batch_size, verbose
    )
    known_q, known_pred, known_metrics, known_cm = evaluate_binary_split(
        model,
        data.X_known_test_scaled,
        data.y_known_test,
        "Known Test",
        batch_size,
        verbose,
    )
    zero_q, zero_pred, zero_metrics = evaluate_zero_day_overall(
        model, data.X_zero_day_scaled, batch_size, verbose
    )
    per_attack = evaluate_per_attack(
        data.zero_day_attack,
        data.y_zero_day,
        zero_pred,
    )

    metrics_df = pd.DataFrame([valid_metrics, known_metrics])
    return {
        "metrics_df": metrics_df,
        "validation_metrics": valid_metrics,
        "known_test_metrics": known_metrics,
        "zero_day_metrics": zero_metrics,
        "per_attack_df": per_attack,
        "validation_cm": valid_cm,
        "known_test_cm": known_cm,
        "known_test_q_values": known_q,
        "known_test_predictions": known_pred,
        "zero_day_q_values": zero_q,
        "zero_day_predictions": zero_pred,
    }


def compare_per_attack(
    tables: dict[str, pd.DataFrame],
    rate_column: str = "Detection Rate (%)",
) -> pd.DataFrame:
    """Side-by-side per-attack detection rates for named runs."""

    merged = None
    for name, table in tables.items():
        piece = table[["Attack", rate_column]].rename(
            columns={rate_column: f"{name}_{rate_column}"}
        )
        merged = piece if merged is None else merged.merge(piece, on="Attack", how="outer")
    return merged
