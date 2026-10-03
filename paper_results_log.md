# Paper Results Log


## Per-family report — 2026-10-03 09:48 (multiseed sweep, seed coverage up to n=8)

Each zero-day family is scored Precision/Recall/F1 against the SAME shared known-test benign pool (see `evaluate_family_breakdown` in `agent/evaluation/metrics.py`), so Precision is comparable across families and configs. Full detail (including per-known-attack-class breakdown) is in `experiments/MULTISEED/per_family_summary.csv` / `.json`; this section is a zero-day-family summary snapshot, dated so later runs don't silently overwrite it.
### Zero-day per-family Detection Rate / Precision / F1 (mean +/- std, feature_set=ports)

| Config | Shellcode | Brute Force | Theft | ransomware | Backdoor |
|---|---|---|---|---|---|
| B0 | DR 0.5+/-0.3 / P 0.0 / F1 0.0 (n=8) | DR 88.1+/-24.3 / P 86.0 / F1 85.1 (n=8) | DR 35.2+/-18.3 / P 1.8 / F1 3.4 (n=8) | DR 86.9+/-31.8 / P 0.3 / F1 0.6 (n=8) | DR 40.9+/-25.5 / P 14.1 / F1 20.8 (n=8) |
| B1 | DR 0.6+/-0.3 / P 0.0 / F1 0.0 (n=8) | DR 94.8+/-9.4 / P 85.6 / F1 89.8 (n=8) | DR 19.9+/-22.4 / P 0.8 / F1 1.5 (n=8) | DR 88.1+/-30.5 / P 0.3 / F1 0.5 (n=8) | DR 43.8+/-22.9 / P 14.7 / F1 22.0 (n=8) |
| B2 | DR 1.0+/-0.4 / P 0.0 / F1 0.0 (n=8) | DR 97.4+/-3.2 / P 84.9 / F1 90.7 (n=8) | DR 23.0+/-26.1 / P 0.8 / F1 1.6 (n=8) | DR 97.8+/-2.4 / P 0.3 / F1 0.6 (n=8) | DR 47.1+/-18.3 / P 15.1 / F1 22.7 (n=8) |
| B3 | DR 0.3+/-0.2 / P 0.0 / F1 0.0 (n=8) | DR 83.8+/-24.2 / P 87.9 / F1 84.0 (n=8) | DR 22.8+/-25.6 / P 1.3 / F1 2.4 (n=8) | DR 96.7+/-2.7 / P 0.5 / F1 0.9 (n=8) | DR 23.5+/-11.1 / P 11.7 / F1 15.4 (n=8) |
| B4 | DR 0.7+/-0.4 / P 0.0 / F1 0.0 (n=8) | DR 94.4+/-11.1 / P 86.1 / F1 89.7 (n=8) | DR 31.1+/-22.1 / P 1.3 / F1 2.5 (n=8) | DR 96.4+/-2.5 / P 0.3 / F1 0.6 (n=8) | DR 44.0+/-24.5 / P 15.1 / F1 22.2 (n=8) |
| B5 | DR 1.0+/-0.6 / P 0.0 / F1 0.0 (n=8) | DR 97.5+/-2.7 / P 83.7 / F1 90.1 (n=8) | DR 17.9+/-25.2 / P 0.6 / F1 1.1 (n=8) | DR 95.3+/-11.5 / P 0.2 / F1 0.5 (n=8) | DR 58.4+/-26.5 / P 16.6 / F1 25.7 (n=8) |
| B6 | DR 0.5+/-0.2 / P 0.0 / F1 0.0 (n=8) | DR 98.3+/-0.6 / P 87.5 / F1 92.6 (n=8) | DR 55.7+/-9.2 / P 2.5 / F1 4.8 (n=8) | DR 96.5+/-2.4 / P 0.3 / F1 0.7 (n=8) | DR 28.0+/-12.8 / P 11.2 / F1 16.0 (n=8) |
| B7 | DR 5.8+/-3.8 / P 0.1 / F1 0.1 (n=8) | DR 99.3+/-0.4 / P 67.5 / F1 80.3 (n=8) | DR 40.5+/-22.0 / P 0.5 / F1 1.1 (n=8) | DR 99.3+/-0.0 / P 0.1 / F1 0.2 (n=8) | DR 98.3+/-0.1 / P 11.9 / F1 21.2 (n=8) |
| B8 | DR 6.6+/-4.8 / P 0.1 / F1 0.2 (n=8) | DR 90.8+/-23.4 / P 68.5 / F1 77.6 (n=8) | DR 48.5+/-16.4 / P 0.8 / F1 1.6 (n=8) | DR 95.4+/-2.5 / P 0.1 / F1 0.2 (n=8) | DR 98.1+/-0.2 / P 14.3 / F1 24.9 (n=8) |
| B9 | DR 7.2+/-3.7 / P 0.1 / F1 0.2 (n=8) | DR 99.1+/-0.2 / P 70.2 / F1 82.2 (n=8) | DR 56.0+/-12.6 / P 0.9 / F1 1.7 (n=8) | DR 87.7+/-19.9 / P 0.1 / F1 0.2 (n=8) | DR 98.1+/-0.2 / P 13.3 / F1 23.3 (n=8) |
| B10 | DR 5.8+/-3.8 / P 0.1 / F1 0.1 (n=8) | DR 99.1+/-0.4 / P 71.8 / F1 83.2 (n=8) | DR 47.5+/-14.1 / P 0.8 / F1 1.6 (n=8) | DR 97.4+/-2.5 / P 0.1 / F1 0.2 (n=8) | DR 98.2+/-0.2 / P 14.3 / F1 24.8 (n=8) |
| B11 | DR 1.2+/-0.4 / P 0.0 / F1 0.0 (n=8) | DR 98.7+/-0.5 / P 79.8 / F1 88.3 (n=8) | DR 43.7+/-14.6 / P 1.1 / F1 2.2 (n=8) | DR 97.4+/-2.4 / P 0.2 / F1 0.4 (n=8) | DR 62.5+/-25.5 / P 14.2 / F1 23.1 (n=8) |
| RF | DR 53.9+/-0.4 / P 25.8 / F1 34.9 (n=8) | DR 66.3+/-0.0 / P 98.9 / F1 79.4 (n=8) | DR 76.9+/-0.3 / P 40.9 / F1 53.4 (n=8) | DR 0.0+/-0.0 / P 0.0 / F1 0.0 (n=8) | DR 45.9+/-0.0 / P 80.5 / F1 58.5 (n=8) |
| XGB | DR 53.6+/-0.8 / P 23.5 / F1 32.6 (n=8) | DR 66.3+/-0.0 / P 98.8 / F1 79.3 (n=8) | DR 76.6+/-0.7 / P 38.0 / F1 50.8 (n=8) | DR 0.3+/-0.4 / P 0.0 / F1 0.0 (n=8) | DR 34.7+/-11.8 / P 71.1 / F1 46.2 (n=8) |
| MLP | DR 17.6+/-9.0 / P 2.4 / F1 4.2 (n=8) | DR 98.7+/-0.3 / P 96.8 / F1 97.7 (n=8) | DR 75.1+/-5.3 / P 13.4 / F1 22.7 (n=8) | DR 0.1+/-0.2 / P 0.0 / F1 0.0 (n=8) | DR 44.0+/-23.1 / P 43.3 / F1 43.2 (n=8) |
| LSTM | DR 11.9+/-2.3 / P 1.8 / F1 3.1 (n=8) | DR 98.7+/-0.3 / P 97.0 / F1 97.8 (n=8) | DR 68.7+/-3.4 / P 13.1 / F1 21.9 (n=8) | DR 4.0+/-3.7 / P 0.1 / F1 0.1 (n=8) | DR 33.2+/-23.6 / P 36.7 / F1 34.3 (n=8) |
