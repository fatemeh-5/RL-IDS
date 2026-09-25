# NIDS-DRL — Zero-Day Detection with Deep Reinforcement Learning

Code root: **`Codes/`**

Stacked-LSTM DQN agents for Benign/Attack detection on **NF-UQ-NIDS**, with five
Zero-Day holdout families. Ablation study **B0–B10**.

Paper: Alam et al., *Adaptive Defense…*, IEEE Access, 2025.

Open this file cleanly in Cursor with **Ctrl+Shift+V**.

---

## Layout

```text
Codes/
├── configs/                 # B0.yaml … B10.yaml
├── agent/                   # RL library (grouped like a classic RL repo)
│   ├── envs/
│   ├── models/
│   ├── buffers/
│   ├── training/
│   ├── evaluation/
│   ├── data/                # constants + dataset pipeline
│   ├── utils/               # config + checkpoints
│   └── runners/             # B0–B10 catalog (CODE, not results)
├── scripts/
│   ├── prepare_data.py
│   └── train.py
├── experiments/             # RESULTS: one CSV per model
├── data/
│   ├── raw/                 # put NF-UQ-NIDS.csv here
│   └── prepared/            # scaler + SMOTE cache
├── notebooks/
│   └── EDA.ipynb
├── docs/
│   └── eda/                 # EDA visual report PDF
├── requirements.txt
└── README.md
```

| Path | Role |
|------|------|
| `configs/` | Experiment hyperparameters |
| `agent/` | How training works |
| `scripts/` | CLI (`prepare_data`, `train`) |
| `experiments/` | Published result CSVs only |
| `data/` | Dataset + cache |
| `notebooks/` | EDA notebook |
| `docs/eda/` | EDA PDF report |

---

## `agent/` package (clean groups)

```text
agent/
├── envs/          # Gymnasium IDS environments
├── models/        # Q-networks (Softmax / Double / Dueling)
├── buffers/       # Uniform replay + PER
├── training/      # Train loops for B0–B10
├── evaluation/    # Known-Test + Zero-Day metrics
├── data/          # Paths, features, leakage-safe pipeline
├── utils/         # YAML config + checkpoint I/O
└── runners/       # Catalog + orchestration for B0–B10
```

| Dir | Contents |
|-----|----------|
| `envs/` | `ids_env.py` — baseline, cost-sensitive, family-aware |
| `models/` | `networks.py` — DQN builders |
| `buffers/` | `replay.py`, `prioritized.py` |
| `training/` | `baseline.py` (B0), `double_dqn.py` (B1–B3), `per.py` (B4–B10), `dueling_helpers.py` |
| `evaluation/` | `metrics.py` |
| `data/` | `constants.py`, `pipeline.py` |
| `utils/` | `config.py`, `checkpointing.py` |
| `runners/` | `catalog.py`, `layout.py`, `runners.py` |

**Do not put result files inside `agent/`.** Results go to top-level `experiments/`.

```python
from agent.models import build_double_dqn
from agent.training import train_per
from agent.runners.catalog import list_experiments
from agent.data import prepare_and_cache
```

---

## Results — single source of truth: `experiments/MULTISEED/`

There is exactly ONE results pipeline. Per-seed cell outputs live under
`experiments/MULTISEED/runs/<drl|supervised>/<config>/<ports|noports>/seed<N>/`
(one `METRICS_ROW.json` + `results/` per cell). Everything else in
`experiments/MULTISEED/` is derived from those cells and safe to regenerate:

```text
experiments/MULTISEED/
├── runs/                              # one dir per (config, feature_set, seed) cell — source of truth
├── sweep_log.jsonl                    # one line per cell attempt (ok/failed), from run_multiseed_sweep.py
├── aggregate_results.csv              # scripts/aggregate_multiseed.py: long table, one row per cell
├── summary_stats.csv, significance_tests.csv, table_main.tex
│                                       # scripts/multiseed_stats.py: mean/std/CI + paired Wilcoxon/t-test
├── per_family_precision_recall_f1.csv # scripts/report_per_family.py: long, all seeds
└── per_family_summary.csv / .json     # scripts/report_per_family.py: mean/std per (config, feature_set, family)
```

`paper_results_log.md` (repo root) gets a dated, versioned section appended
by `scripts/report_per_family.py` each time it's run — never overwritten.

There used to be a second, older per-model CSV pipeline
(`experiments/B0_zero_day_per_attack_metrics.csv`, etc., written by
`scripts/train.py <ID>` single runs). It was removed because it silently
diverged from the multi-seed results and one of its files was stale. If you
need a single, non-swept run for a quick check, `scripts/train.py <ID>` still
works and writes to `experiments/<family_dir>/`, but for anything going in
the paper, use the `MULTISEED` pipeline above.

To rebuild everything from the sweep's raw cells:
```bash
python scripts/aggregate_multiseed.py     # -> aggregate_results.csv
python scripts/multiseed_stats.py         # -> summary_stats.csv, significance_tests.csv, STATS_NOTES.md
python scripts/report_per_family.py       # -> per_family_*.csv/json, appends to paper_results_log.md
```

---

## Quick start

```bash
cd Codes
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# required:
#   data/raw/NF-UQ-NIDS.csv

python3 scripts/prepare_data.py --force
python3 scripts/train.py --list
python3 scripts/train.py B10
python3 scripts/train.py --all
```

Smoke / resume:

```bash
python3 scripts/train.py B0 --episodes 2
python3 scripts/train.py B4 --mode resume
```

---

## Experiments B0–B10

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

Configs: `configs/B*.yaml`  
Registry: `agent/runners/catalog.py`

---

## Parameters

1. Global / data → `agent/data/constants.py`
2. Per experiment → `configs/B*.yaml`
3. Networks → `agent/models/networks.py`

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
| Dataset missing | Place CSV at `data/raw/NF-UQ-NIDS.csv` |
| Cache missing | `python3 scripts/prepare_data.py --force` |
| Import errors | Run commands from `Codes/` |
