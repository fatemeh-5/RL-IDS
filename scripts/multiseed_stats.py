#!/usr/bin/env python3
"""Publication-grade aggregate stats + significance tests over multi-seed runs.

Reads ONLY experiments/MULTISEED/aggregate_results.csv (long format: one row
per config x feature_set x model_type x seed). Writes:
  - experiments/MULTISEED/summary_stats.csv    (mean/std/95% CI per group)
  - experiments/MULTISEED/significance_tests.csv (paired Wilcoxon + t-test)
  - experiments/MULTISEED/table_main.tex       (booktabs headline table)
  - STATS_NOTES.md                             (plain-English readout)

Deterministic; never crashes on missing/partial groups (<2 seeds -> mean only,
std/CI/tests reported as N/A).

Methodology notes (also restated in STATS_NOTES.md):
  - Seeds are matched across configs/feature_sets, so every comparison below
    is a PAIRED comparison on the intersection of seeds present in both
    groups (inner join on seed).
  - Primary test: Wilcoxon signed-rank (nonparametric, appropriate for the
    small seed count). Secondary: paired t-test.
  - Holm-Bonferroni correction is applied within each test family (all
    Wilcoxon p-values together; all paired-t p-values together) across every
    comparison x metric run in this script — not per-comparison — since
    that's the actual family of hypotheses a reader would need corrected
    together to control the overall false-discovery risk.
  - Unless a comparison explicitly varies feature_set (the two ports-vs-
    noports rows), all comparisons use feature_set="ports" (the default /
    main-results feature set).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MULTISEED_DIR = ROOT / "experiments" / "MULTISEED"
AGG_CSV = MULTISEED_DIR / "aggregate_results.csv"
SUMMARY_CSV = MULTISEED_DIR / "summary_stats.csv"
SIG_CSV = MULTISEED_DIR / "significance_tests.csv"
TABLE_TEX = MULTISEED_DIR / "table_main.tex"
NOTES_MD = ROOT / "STATS_NOTES.md"

METRIC_COLUMNS = [
    "KT_Accuracy", "KT_Precision", "KT_Recall", "KT_F1", "KT_ROC_AUC",
    "Weighted_ZeroDay", "Macro_ZeroDay",
    "Shellcode", "Brute Force", "Theft", "ransomware", "Backdoor",
]
SIG_METRICS = ["Macro_ZeroDay", "Weighted_ZeroDay", "KT_ROC_AUC"]
TABLE_ROWS = ["RF", "XGB", "MLP", "LSTM", "B1", "B4", "B6", "B8", "B10"]
TABLE_COLS = ["KT_ROC_AUC", "Weighted_ZeroDay", "Macro_ZeroDay"]

NA = "N/A"


# --------------------------------------------------------------------------- #
# Step 1: summary_stats.csv                                                   #
# --------------------------------------------------------------------------- #
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["config", "feature_set", "model_type"]
    for keys, group in df.groupby(group_cols, sort=True):
        config, feature_set, model_type = keys
        n = len(group)
        row = {"config": config, "feature_set": feature_set, "model_type": model_type, "n_seeds": n}
        for metric in METRIC_COLUMNS:
            vals = group[metric].dropna().to_numpy(dtype=float)
            n_valid = len(vals)
            mean = float(np.mean(vals)) if n_valid else float("nan")
            row[f"{metric}_mean"] = mean
            if n_valid < 2:
                row[f"{metric}_std"] = NA
                row[f"{metric}_ci95"] = NA
            else:
                std = float(np.std(vals, ddof=1))
                se = std / np.sqrt(n_valid)
                tcrit = float(stats.t.ppf(0.975, df=n_valid - 1))
                row[f"{metric}_std"] = std
                row[f"{metric}_ci95"] = tcrit * se
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Step 2: significance_tests.csv                                              #
# --------------------------------------------------------------------------- #
def get_group(df: pd.DataFrame, config: str, feature_set: str, model_type: str) -> pd.Series | None:
    sub = df[
        (df["config"] == config)
        & (df["feature_set"] == feature_set)
        & (df["model_type"] == model_type)
    ]
    if sub.empty:
        return None
    return sub


def paired_values(df: pd.DataFrame, a_key: tuple, b_key: tuple, metric: str):
    """a_key/b_key = (config, feature_set, model_type). Returns aligned arrays by seed."""
    ga = get_group(df, *a_key)
    gb = get_group(df, *b_key)
    if ga is None or gb is None:
        return np.array([]), np.array([]), []
    ga = ga.set_index("seed")[metric]
    gb = gb.set_index("seed")[metric]
    joined = pd.concat([ga, gb], axis=1, keys=["a", "b"], join="inner").dropna()
    return joined["a"].to_numpy(dtype=float), joined["b"].to_numpy(dtype=float), list(joined.index)


def rank_biserial(diffs: np.ndarray) -> float | str:
    """Matched-pairs rank-biserial effect size for the Wilcoxon signed-rank test."""
    nz = diffs[diffs != 0]
    n = len(nz)
    if n == 0:
        return NA
    ranks = stats.rankdata(np.abs(nz))
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    total = w_pos + w_neg
    if total == 0:
        return NA
    return float((w_pos - w_neg) / total)


def cohens_d_paired(diffs: np.ndarray) -> float | str:
    if len(diffs) < 2:
        return NA
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return NA
    return float(np.mean(diffs) / sd)


def run_pair_test(a: np.ndarray, b: np.ndarray) -> dict:
    """One comparison, one metric: paired Wilcoxon + paired t-test, raw (unadjusted)."""
    n = len(a)
    out = {
        "n_pairs": n,
        "mean_diff": float(np.mean(a - b)) if n else NA,
        "wilcoxon_stat": NA, "wilcoxon_p": NA, "wilcoxon_rank_biserial": NA,
        "ttest_stat": NA, "ttest_p": NA, "cohens_d": NA,
    }
    if n < 2:
        return out
    diffs = a - b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            w_stat, w_p = stats.wilcoxon(diffs)
            out["wilcoxon_stat"] = float(w_stat)
            out["wilcoxon_p"] = float(w_p)
            out["wilcoxon_rank_biserial"] = rank_biserial(diffs)
        except ValueError:
            pass  # all-zero differences etc. -> stays N/A

        try:
            t_stat, t_p = stats.ttest_rel(a, b)
            if np.isfinite(t_stat) and np.isfinite(t_p):
                out["ttest_stat"] = float(t_stat)
                out["ttest_p"] = float(t_p)
                out["cohens_d"] = cohens_d_paired(diffs)
        except Exception:
            pass

    return out


def holm_bonferroni(pvals: list[float | str]) -> list[float | str]:
    """Holm-Bonferroni step-down correction; N/A entries pass through untouched."""
    idx_valid = [i for i, p in enumerate(pvals) if p != NA]
    m = len(idx_valid)
    adjusted = list(pvals)
    if m == 0:
        return adjusted
    order = sorted(idx_valid, key=lambda i: pvals[i])
    running_max = 0.0
    for rank, i in enumerate(order):
        factor = m - rank
        val = min(1.0, pvals[i] * factor)
        running_max = max(running_max, val)
        adjusted[i] = running_max
    return adjusted


def build_significance(df: pd.DataFrame) -> pd.DataFrame:
    fs = "ports"

    # Determine best supervised baseline (RF vs XGB) by mean Macro_ZeroDay, ports.
    best_sup = None
    best_sup_mean = -np.inf
    for model in ["RF", "XGB"]:
        g = get_group(df, model, fs, "supervised")
        if g is not None and g["Macro_ZeroDay"].notna().any():
            m = g["Macro_ZeroDay"].mean()
            if m > best_sup_mean:
                best_sup_mean = m
                best_sup = model
    if best_sup is None:
        best_sup = "RF"  # fallback label; comparison rows will just come back empty

    comparisons = [
        ("B10_vs_best_supervised", ("B10", fs, "drl"), (best_sup, fs, "supervised"),
         f"B10 vs best supervised baseline ({best_sup})"),
        ("B10_vs_B8", ("B10", fs, "drl"), ("B8", fs, "drl"), "B10 vs B8"),
        ("B0_vs_B1", ("B1", fs, "drl"), ("B0", fs, "drl"), "B1 vs B0 (ablation jump)"),
        ("B3_vs_B4", ("B4", fs, "drl"), ("B3", fs, "drl"), "B4 vs B3 (ablation jump)"),
        ("B6_vs_B8", ("B8", fs, "drl"), ("B6", fs, "drl"), "B8 vs B6 (ablation jump)"),
        ("B9_vs_B10", ("B10", fs, "drl"), ("B9", fs, "drl"), "B10 vs B9 (ablation jump)"),
        ("ports_vs_noports_B8", ("B8", "ports", "drl"), ("B8", "noports", "drl"), "B8: ports vs noports"),
        ("ports_vs_noports_B10", ("B10", "ports", "drl"), ("B10", "noports", "drl"), "B10: ports vs noports"),
    ]

    rows = []
    for comp_id, a_key, b_key, label in comparisons:
        for metric in SIG_METRICS:
            a, b, seeds = paired_values(df, a_key, b_key, metric)
            result = run_pair_test(a, b)
            rows.append({
                "comparison": comp_id,
                "label": label,
                "group_a": f"{a_key[0]}|{a_key[1]}|{a_key[2]}",
                "group_b": f"{b_key[0]}|{b_key[1]}|{b_key[2]}",
                "metric": metric,
                "n_pairs": result["n_pairs"],
                "seeds_used": ",".join(str(s) for s in seeds),
                "mean_diff_a_minus_b": result["mean_diff"],
                "wilcoxon_stat": result["wilcoxon_stat"],
                "wilcoxon_p_raw": result["wilcoxon_p"],
                "wilcoxon_rank_biserial": result["wilcoxon_rank_biserial"],
                "ttest_stat": result["ttest_stat"],
                "ttest_p_raw": result["ttest_p"],
                "cohens_d": result["cohens_d"],
            })

    sig_df = pd.DataFrame(rows)
    sig_df["wilcoxon_p_holm"] = holm_bonferroni(sig_df["wilcoxon_p_raw"].tolist())
    sig_df["ttest_p_holm"] = holm_bonferroni(sig_df["ttest_p_raw"].tolist())

    col_order = [
        "comparison", "label", "group_a", "group_b", "metric", "n_pairs", "seeds_used",
        "mean_diff_a_minus_b",
        "wilcoxon_stat", "wilcoxon_p_raw", "wilcoxon_p_holm", "wilcoxon_rank_biserial",
        "ttest_stat", "ttest_p_raw", "ttest_p_holm", "cohens_d",
    ]
    return sig_df[col_order]


# --------------------------------------------------------------------------- #
# Step 3: table_main.tex                                                      #
# --------------------------------------------------------------------------- #
def fmt_cell(mean, std, n) -> str:
    if pd.isna(mean):
        return "--"
    if std == NA or not isinstance(std, (int, float)) or pd.isna(std):
        return f"{mean:.2f} (n={n})"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def build_table(summary_df: pd.DataFrame) -> str:
    fs = "ports"
    lookup = summary_df.set_index(["config", "feature_set"])

    header = " & ".join(["Model"] + [c.replace("_", r"\_") for c in TABLE_COLS])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Headline multi-seed results (mean $\pm$ std across matched seeds, ports feature set).}",
        r"\label{tab:multiseed-main}",
        r"\begin{tabular}{l" + "c" * len(TABLE_COLS) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for model in TABLE_ROWS:
        key = (model, fs)
        cells = [model]
        if key not in lookup.index:
            cells.extend(["--"] * len(TABLE_COLS))
        else:
            row = lookup.loc[key]
            n = int(row["n_seeds"])
            for col in TABLE_COLS:
                mean = row.get(f"{col}_mean", float("nan"))
                std = row.get(f"{col}_std", NA)
                cells.append(fmt_cell(mean, std, n))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Step 4: STATS_NOTES.md                                                      #
# --------------------------------------------------------------------------- #
def direction_word(mean_diff, metric_higher_is_better=True) -> str:
    if mean_diff == NA or (isinstance(mean_diff, float) and pd.isna(mean_diff)):
        return "no comparable data"
    if mean_diff > 0:
        return "higher" if metric_higher_is_better else "lower"
    if mean_diff < 0:
        return "lower" if metric_higher_is_better else "higher"
    return "identical"


def build_notes(sig_df: pd.DataFrame, n_seeds_planned: int) -> str:
    lines = [
        "# Multi-seed statistics notes",
        "",
        f"Generated from `experiments/MULTISEED/aggregate_results.csv` "
        f"(up to {n_seeds_planned} matched seeds per group; groups with fewer "
        "seeds are reported with N/A significance).",
        "",
        "Methodology: paired Wilcoxon signed-rank (primary) and paired t-test "
        "(secondary) on the seed-matched intersection for each comparison; "
        "Holm-Bonferroni correction applied within each test family (all "
        "Wilcoxon p-values together, all paired-t p-values together) across "
        "every comparison x metric below. \"Significant\" = Wilcoxon Holm-adjusted "
        "p < 0.05.",
        "",
    ]

    # Power caveat, parameterized by the ACTUAL seed count in this run (not a
    # hardcoded assumption) — the exact two-sided Wilcoxon signed-rank test's
    # smallest achievable p-value for n matched pairs (all-same-sign, no ties)
    # is 2^(1-n).
    if n_seeds_planned >= 2:
        p_min = 2.0 ** (1 - n_seeds_planned)
        denom = int(round(1 / p_min))
        lines += [
            f"**Power caveat:** with n={n_seeds_planned} matched seeds, the exact "
            f"two-sided Wilcoxon signed-rank test's smallest achievable p-value is "
            f"1/{denom} ~= {p_min:.4f}; Holm-Bonferroni across this family's tests "
            "inflates that floor well above 0.05, so Wilcoxon alone cannot certify "
            "significance here even for a perfectly consistent effect across all "
            "seeds. Treat the paired-t p-value (reported alongside, uncorrected "
            "effect estimate via Cohen's d) as corroborating evidence, and treat "
            "\"not significant\" readings below as \"underpowered at this seed "
            "count,\" not as \"no effect.\"",
            "",
        ]
    else:
        lines += [
            f"**Power caveat:** only n={n_seeds_planned} seed(s) available — the "
            "Wilcoxon signed-rank test requires at least 2 matched pairs, so "
            "significance columns below are N/A throughout.",
            "",
        ]

    for comp_id, comp_df in sig_df.groupby("comparison", sort=False):
        label = comp_df["label"].iloc[0]
        lines.append(f"## {label}")
        for _, r in comp_df.iterrows():
            metric = r["metric"]
            n = r["n_pairs"]
            if n < 2:
                lines.append(f"- **{metric}**: insufficient matched seeds (n={n}) — N/A.")
                continue
            md = r["mean_diff_a_minus_b"]
            direction = direction_word(md)
            w_p = r["wilcoxon_p_holm"]
            sig = isinstance(w_p, float) and w_p < 0.05
            sig_word = "significant" if sig else "not significant"
            w_p_str = f"{w_p:.4f}" if isinstance(w_p, float) else NA
            t_p = r["ttest_p_holm"]
            t_p_str = f"{t_p:.4f}" if isinstance(t_p, float) else NA
            lines.append(
                f"- **{metric}** (n={n}): group A {direction} than group B by "
                f"{abs(md):.2f} pts on average — {sig_word} after Holm correction "
                f"(Wilcoxon p={w_p_str}, paired-t p={t_p_str})."
            )
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    if not AGG_CSV.exists():
        print(f"Missing input: {AGG_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(AGG_CSV)
    required = {"config", "feature_set", "model_type", "seed", *METRIC_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"aggregate_results.csv missing columns: {sorted(missing)}")

    n_seeds_planned = int(df["seed"].nunique())

    summary_df = build_summary(df)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV} ({len(summary_df)} rows)")

    sig_df = build_significance(df)
    sig_df.to_csv(SIG_CSV, index=False)
    print(f"Wrote {SIG_CSV} ({len(sig_df)} rows)")

    table_tex = build_table(summary_df)
    TABLE_TEX.write_text(table_tex, encoding="utf-8")
    print(f"Wrote {TABLE_TEX}")

    notes_md = build_notes(sig_df, n_seeds_planned)
    NOTES_MD.write_text(notes_md, encoding="utf-8")
    print(f"Wrote {NOTES_MD}")


if __name__ == "__main__":
    main()
