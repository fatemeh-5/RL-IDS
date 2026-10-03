# Multi-seed statistics notes

Generated from `experiments/MULTISEED/aggregate_results.csv` (up to 8 matched seeds per group; groups with fewer seeds are reported with N/A significance).

Methodology: paired Wilcoxon signed-rank (primary) and paired t-test (secondary) on the seed-matched intersection for each comparison; Holm-Bonferroni correction applied within each test family (all Wilcoxon p-values together, all paired-t p-values together) across every comparison x metric below. "Significant" = Wilcoxon Holm-adjusted p < 0.05.

**Power caveat:** with n=8 matched seeds, the exact two-sided Wilcoxon signed-rank test's smallest achievable p-value is 1/128 ~= 0.0078; Holm-Bonferroni across this family's tests inflates that floor well above 0.05, so Wilcoxon alone cannot certify significance here even for a perfectly consistent effect across all seeds. Treat the paired-t p-value (reported alongside, uncorrected effect estimate via Cohen's d) as corroborating evidence, and treat "not significant" readings below as "underpowered at this seed count," not as "no effect."

## B10 vs best supervised baseline (RF)
- **Macro_ZeroDay** (n=8): group A higher than group B by 20.97 pts on average — not significant after Holm correction (Wilcoxon p=0.1406, paired-t p=0.0000).
- **Weighted_ZeroDay** (n=8): group A higher than group B by 33.24 pts on average — not significant after Holm correction (Wilcoxon p=0.1406, paired-t p=0.0000).
- **KT_ROC_AUC** (n=8): group A lower than group B by 6.15 pts on average — not significant after Holm correction (Wilcoxon p=0.1406, paired-t p=0.0001).

## B10 vs B8
- **Macro_ZeroDay** (n=8): group A higher than group B by 1.68 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **Weighted_ZeroDay** (n=8): group A higher than group B by 7.64 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **KT_ROC_AUC** (n=8): group A lower than group B by 0.66 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).

## B1 vs B0 (ablation jump)
- **Macro_ZeroDay** (n=8): group A lower than group B by 0.87 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **Weighted_ZeroDay** (n=8): group A higher than group B by 6.31 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **KT_ROC_AUC** (n=8): group A lower than group B by 10.41 pts on average — not significant after Holm correction (Wilcoxon p=0.1406, paired-t p=0.0085).

## B4 vs B3 (ablation jump)
- **Macro_ZeroDay** (n=8): group A higher than group B by 7.90 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **Weighted_ZeroDay** (n=8): group A higher than group B by 11.08 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **KT_ROC_AUC** (n=8): group A higher than group B by 3.00 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).

## B8 vs B6 (ablation jump)
- **Macro_ZeroDay** (n=8): group A higher than group B by 12.12 pts on average — not significant after Holm correction (Wilcoxon p=0.1406, paired-t p=0.0049).
- **Weighted_ZeroDay** (n=8): group A lower than group B by 2.68 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **KT_ROC_AUC** (n=8): group A higher than group B by 2.01 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).

## B10 vs B9 (ablation jump)
- **Macro_ZeroDay** (n=8): group A lower than group B by 0.02 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **Weighted_ZeroDay** (n=8): group A lower than group B by 0.09 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **KT_ROC_AUC** (n=8): group A lower than group B by 0.97 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).

## B8: ports vs noports
- **Macro_ZeroDay**: insufficient matched seeds (n=0) — N/A.
- **Weighted_ZeroDay**: insufficient matched seeds (n=0) — N/A.
- **KT_ROC_AUC**: insufficient matched seeds (n=0) — N/A.

## B10: ports vs noports
- **Macro_ZeroDay**: insufficient matched seeds (n=0) — N/A.
- **Weighted_ZeroDay**: insufficient matched seeds (n=0) — N/A.
- **KT_ROC_AUC**: insufficient matched seeds (n=0) — N/A.
