# Multi-seed statistics notes

Generated from `experiments/MULTISEED/aggregate_results.csv` (up to 2 matched seeds per group; groups with fewer seeds are reported with N/A significance).

Methodology: paired Wilcoxon signed-rank (primary) and paired t-test (secondary) on the seed-matched intersection for each comparison; Holm-Bonferroni correction applied within each test family (all Wilcoxon p-values together, all paired-t p-values together) across every comparison x metric below. "Significant" = Wilcoxon Holm-adjusted p < 0.05.

**Power caveat:** with n=2 matched seeds, the exact two-sided Wilcoxon signed-rank test's smallest achievable p-value is 1/2 ~= 0.5000; Holm-Bonferroni across this family's tests inflates that floor well above 0.05, so Wilcoxon alone cannot certify significance here even for a perfectly consistent effect across all seeds. Treat the paired-t p-value (reported alongside, uncorrected effect estimate via Cohen's d) as corroborating evidence, and treat "not significant" readings below as "underpowered at this seed count," not as "no effect."

## B10 vs best supervised baseline (RF)
- **Macro_ZeroDay** (n=2): group A higher than group B by 22.36 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=0.7883).
- **Weighted_ZeroDay** (n=2): group A higher than group B by 32.77 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=0.0145).
- **KT_ROC_AUC** (n=2): group A lower than group B by 4.26 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).

## B10 vs B8
- **Macro_ZeroDay** (n=2): group A higher than group B by 5.71 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **Weighted_ZeroDay** (n=2): group A higher than group B by 1.28 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **KT_ROC_AUC** (n=2): group A higher than group B by 0.44 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).

## B1 vs B0 (ablation jump)
- **Macro_ZeroDay** (n=2): group A lower than group B by 0.02 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **Weighted_ZeroDay** (n=2): group A lower than group B by 0.63 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **KT_ROC_AUC** (n=2): group A lower than group B by 3.13 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).

## B4 vs B3 (ablation jump)
- **Macro_ZeroDay** (n=2): group A lower than group B by 1.33 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **Weighted_ZeroDay** (n=2): group A lower than group B by 6.10 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **KT_ROC_AUC** (n=2): group A higher than group B by 1.61 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).

## B8 vs B6 (ablation jump)
- **Macro_ZeroDay** (n=2): group A higher than group B by 8.26 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **Weighted_ZeroDay** (n=2): group A higher than group B by 1.41 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **KT_ROC_AUC** (n=2): group A higher than group B by 1.46 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).

## B10 vs B9 (ablation jump)
- **Macro_ZeroDay** (n=2): group A lower than group B by 2.09 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **Weighted_ZeroDay** (n=2): group A lower than group B by 1.02 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **KT_ROC_AUC** (n=2): group A higher than group B by 0.53 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).

## B8: ports vs noports
- **Macro_ZeroDay** (n=2): group A higher than group B by 15.51 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **Weighted_ZeroDay** (n=2): group A higher than group B by 27.79 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **KT_ROC_AUC** (n=2): group A higher than group B by 6.65 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=0.9190).

## B10: ports vs noports
- **Macro_ZeroDay** (n=2): group A higher than group B by 41.50 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
- **Weighted_ZeroDay** (n=2): group A higher than group B by 90.80 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=0.0268).
- **KT_ROC_AUC** (n=2): group A higher than group B by 16.08 pts on average — not significant after Holm correction (Wilcoxon p=1.0000, paired-t p=1.0000).
