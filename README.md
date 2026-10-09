# NIDS-DRL — Zero-Day Detection with Deep Reinforcement Learning

Stacked-LSTM DQN agents for Benign/Attack detection on **NF-UQ-NIDS**, with five
Zero-Day holdout families (Shellcode, Brute Force, Theft, ransomware, Backdoor).
The paper reports the ablation study **B0–B10**, evaluated with an 8-seed sweep,
paired significance tests, and per-family precision/recall/F1.

This work builds on:

> K. Alam, M. F. Monir, M. J. Hossain, M. S. Uddin, and M. T. Habib,
> "Adaptive Defense: Zero-Day Attack Detection in NIDS With Deep Reinforcement
> Learning," *IEEE Access*, vol. 13, 2025. doi:[10.1109/ACCESS.2025.3585445](https://doi.org/10.1109/ACCESS.2025.3585445)

All commands below are run from the repository root.

---

## Requirements

| | |
|---|---|
| Python | 3.12 (tested with 3.12.10) |
| Packages | pinned in `requirements.txt` (TensorFlow 2.21, scikit-learn 1.9, imbalanced-learn 0.14, gymnasium 1.3, …) |
| OS | Developed on Windows 11; the Python code is OS-independent |
| Hardware used | AMD Ryzen 9 3900X (12 cores), 128 GB RAM, **CPU only** (no GPU) |
| Runtime (on that machine) | DRL cell (one config × one seed): median ~65 min, range 8–85 min. Supervised cell: median ~9 min. Full DRL sweep (12 configs × 8 seeds = 96 cells): ~105 h |

---

## Dataset

The dataset is **not** included in this repository.

- **Dataset:** NF-UQ-NIDS (v1, the 8 NetFlow features + IPs/ports, with `Attack`
  and `Dataset` columns): ~12M flows merged from UNSW-NB15, BoT-IoT, ToN-IoT and
  CSE-CIC-IDS2018.
- **Download:** University of Queensland, *Machine Learning-Based NIDS Datasets*:
  <https://staff.itee.uq.edu.au/marius/NIDS_datasets/> (also listed at
  <https://www.cyber.uq.edu.au/node/824>).
- **License / terms:** set by the dataset authors. Academic research use is
  permitted provided the papers below are cited. For any other use, check the
  terms on the download page or contact the authors. This repository does not
  redistribute the data.
- **Citation:** M. Sarhan, S. Layeghy, N. Moustafa, and M. Portmann, "NetFlow
  Datasets for Machine Learning-Based Network Intrusion Detection Systems," in
  *Big Data Technologies and Applications (BDTA 2020)*, LNICST vol. 371,
  Springer, 2021. doi:[10.1007/978-3-030-72802-1_9](https://doi.org/10.1007/978-3-030-72802-1_9)

Place the CSV at:

```text
data/raw/NF-UQ-NIDS.csv
```

`data/` is git-ignored, so create `data/raw/` yourself.

---

## Quick start

Shell conventions: every `python ...` command below works unchanged in both
bash and PowerShell (forward slashes are fine on Windows). Only lines that
differ between shells are given twice and labelled.

```bash
# --- Linux / macOS (bash) ---
python3.12 -m venv .venv
source .venv/bin/activate
```

```powershell
# --- Windows (PowerShell) ---
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Then, in either shell:

```bash
pip install -r requirements.txt

python scripts/prepare_data.py --force     # scaler + SMOTE cache -> data/prepared/
python scripts/train.py --list
python scripts/train.py B10                # single run, quick check
python scripts/train.py B0 --episodes 2    # smoke test
python scripts/train.py B4 --mode resume   # resume from checkpoint
```

### Reproducing the paper results (multi-seed sweep)

```bash
python scripts/run_multiseed_sweep.py --dry-run   # print the plan
python scripts/run_multiseed_sweep.py             # resumable; safe to kill and re-run
```

The sweep runs seeds 42–49. A cell counts as done when its `METRICS_ROW.json`
exists, so re-running the command picks up where it stopped.

Watching a running sweep:

```bash
python scripts/sweep_status.py --tail 15   # current model/seed/episode, done/total, ETA
```

```bash
# --- Linux / macOS (bash) ---
tail -f -n 20 experiments/MULTISEED/progress.log
```

```powershell
# --- Windows (PowerShell) ---
Get-Content experiments\MULTISEED\progress.log -Wait -Tail 20
```

`scripts/watchdog_sweep.ps1` is an optional Windows-only helper. It relaunches
the sweep after a crash or reboot when registered as a Scheduled Task. On
Linux, use cron/systemd or simply re-run the sweep command.

Rebuilding all derived tables from the raw cells:

```bash
python scripts/aggregate_multiseed.py   # -> aggregate_results.csv
python scripts/multiseed_stats.py       # -> summary_stats.csv, significance_tests.csv, table_main.tex, STATS_NOTES.md
python scripts/report_per_family.py     # -> per_family_*.csv/json, appends to paper_results_log.md
```

---

## Experiments

### Paper ablation: B0–B10

| ID | Idea | Trainer |
|----|------|---------|
| B0 | Softmax DQN baseline | `training/baseline.py` |
| B1 | Double DQN | `training/double_dqn.py` |
| B2 | Episode ε decay | `training/double_dqn.py` |
| B3 | Large uniform replay | `training/double_dqn.py` |
| B4 | PER | `training/per.py` |
| B5 | Cost-sensitive + PER | `training/per.py` |
| B6 | Dueling + PER | `training/per.py` |
| B7 | Family-stratified sampling | `training/per.py` |
| B8 | Hybrid sampling | `training/per.py` |
| B9 | Adaptive hybrid | `training/per.py` |
| B10 | Restrained adaptive | `training/per.py` |

Configs: `configs/B*.yaml`. Registry: `agent/runners/catalog.py`.

Supervised baselines (RF, XGBoost, MLP, LSTM) are run by
`scripts/supervised_baseline.py` as part of the sweep.

### Exploratory extensions (not reported in the paper)

- **B11** (`configs/B11.yaml`, `agent/envs/investigate_env.py`): B10 sampling
  plus a third "investigate" action and a post-alert queue effect. It is part of
  the default sweep and appears in the aggregate CSVs, but it is **not** part of
  the B0–B10 ablation in the paper. To skip it, remove `"B11"` from
  `DRL_CONFIGS` in `scripts/run_multiseed_sweep.py`.
- **Advanced algorithms** (`configs/advanced/`, `agent/training/advanced/`,
  `experiments/ADV_*`): PPO, A2C, Rainbow, C51, QR-DQN, etc. These are
  exploratory and not in the paper.

---

## Layout

```text
.
├── configs/                 # B0.yaml … B10.yaml (paper), B11.yaml + advanced/ (exploratory)
├── agent/                   # RL library
│   ├── envs/                # Gymnasium IDS environments
│   ├── models/              # Q-networks (Softmax / Double / Dueling)
│   ├── buffers/             # Uniform replay + PER
│   ├── training/            # Train loops (+ advanced/)
│   ├── evaluation/          # Known-Test + Zero-Day metrics
│   ├── data/                # constants + leakage-safe dataset pipeline
│   ├── utils/               # config, checkpoints, progress logging
│   └── runners/             # experiment catalog + orchestration (code, not results)
├── scripts/                 # CLI: prepare_data, train, multi-seed sweep, aggregation, stats
├── experiments/
│   └── MULTISEED/           # paper results (derived tables tracked in git)
├── data/                    # git-ignored: raw/NF-UQ-NIDS.csv, prepared/ cache
├── notebooks/EDA.ipynb
├── docs/eda/                # EDA visual report PDF
├── STATS_NOTES.md           # statistical-testing notes (generated)
├── paper_results_log.md     # dated, append-only results log
├── requirements.txt
└── LICENSE
```

**Do not put result files inside `agent/`.** Results go to `experiments/`.

| Dir | Contents |
|-----|----------|
| `agent/envs/` | `ids_env.py` (baseline, cost-sensitive, family-aware), `investigate_env.py` (B11) |
| `agent/models/` | `networks.py` (DQN builders) |
| `agent/buffers/` | `replay.py`, `prioritized.py` |
| `agent/training/` | `baseline.py` (B0), `double_dqn.py` (B1–B3), `per.py` (B4–B11), `dueling_helpers.py` |
| `agent/evaluation/` | `metrics.py` |
| `agent/data/` | `constants.py`, `pipeline.py` |
| `agent/utils/` | `config.py`, `checkpointing.py`, `progress.py` |
| `agent/runners/` | `catalog.py`, `layout.py`, `runners.py` |

Parameters live in three places:

1. Global / data: `agent/data/constants.py`
2. Per experiment: `configs/B*.yaml`
3. Networks: `agent/models/networks.py`

---

## Results: `experiments/MULTISEED/`

There is a single results pipeline. Per-seed cell outputs live under
`experiments/MULTISEED/runs/<drl|supervised>/<config>/<ports|noports>/seed<N>/`
(one `METRICS_ROW.json` + `results/` per cell; git-ignored because of size).
Everything else is derived from those cells and safe to regenerate:

```text
experiments/MULTISEED/
├── runs/                              # one dir per (config, feature_set, seed) cell (source of truth)
├── sweep_log.jsonl                    # one line per cell attempt (ok/failed)
├── progress.log                       # live log: timestamp | model features seed | episode x/N, reward, acc, eps, ETA
├── aggregate_results.csv              # long table, one row per cell
├── summary_stats.csv, significance_tests.csv, table_main.tex
│                                      # mean/std/CI + paired Wilcoxon/t-test
├── per_family_precision_recall_f1.csv # long, all seeds
└── per_family_summary.csv / .json     # mean/std per (config, feature_set, family)
```

Each cell's full TensorFlow output goes to its own
`runs/<kind>/<config>/<features>/seed<N>/log.txt`.

`scripts/train.py <ID>` single runs write to `experiments/<family_dir>/` and are
meant for quick checks only. Use the `MULTISEED` pipeline for anything reported.

---

## EDA materials

| File | Location |
|------|----------|
| Notebook | `notebooks/EDA.ipynb` |
| Visual report PDF | `docs/eda/NF-UQ-NIDS_EDA_Complete_Visual_Report.pdf` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Dataset missing | Place CSV at `data/raw/NF-UQ-NIDS.csv` (see [Dataset](#dataset)) |
| Cache missing | `python scripts/prepare_data.py --force` |
| Import errors | Run commands from the repository root with the venv active |

---

## License

Code: [MIT](LICENSE). The NF-UQ-NIDS dataset is subject to its own terms (see
[Dataset](#dataset)).
