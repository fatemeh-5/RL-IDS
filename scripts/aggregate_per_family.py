from pathlib import Path
import json
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

# This is the MULTISEED directory containing "runs"
ROOT = Path(
    r"C:\Users\Dr. Zarepour\Desktop\RL_IDS\Codes\experiments\MULTISEED"
)

RUNS_DIR = ROOT / "runs"

# Output: ONE file containing aggregated per-attack results
OUTPUT_FILE = ROOT / "per_attack_ports_aggregated.csv"

# Expected Zero-Day attack names
ATTACKS = [
    "Backdoor",
    "Brute Force",
    "Shellcode",
    "Theft",
    "ransomware",
]


# ============================================================
# HELPERS
# ============================================================

def get_model_name(metrics_path: Path) -> str:
    """
    Convert the directory structure into a readable model name.

    Examples:
        runs/drl/B1/ports/seed42/METRICS_ROW.json
            -> B1

        runs/supervised/rf/ports/seed42/METRICS_ROW.json
            -> RF
    """

    parts = metrics_path.parts

    # Find "runs"
    try:
        runs_idx = parts.index("runs")
    except ValueError:
        return metrics_path.parent.name

    model_type = parts[runs_idx + 1]

    if model_type == "drl":
        # runs / drl / B1 / ports / seed42 / METRICS_ROW.json
        config = parts[runs_idx + 2]
        return config

    if model_type == "supervised":
        # runs / supervised / rf / ports / seed42 / METRICS_ROW.json
        model = parts[runs_idx + 2]
        return model.upper()

    return metrics_path.parent.name


def get_seed(metrics_path: Path):
    """
    Extract seed from:
        .../seed42/METRICS_ROW.json
    """
    seed_dir = metrics_path.parent.name

    if seed_dir.startswith("seed"):
        try:
            return int(seed_dir.replace("seed", ""))
        except ValueError:
            pass

    return None


def normalize_attack_name(name: str) -> str:
    """
    Normalize attack names so that minor capitalization differences
    don't create separate rows.
    """

    name = str(name).strip()

    mapping = {
        "backdoor": "Backdoor",
        "brute force": "Brute Force",
        "shellcode": "Shellcode",
        "theft": "Theft",
        "ransomware": "ransomware",
        "Ransomware": "ransomware",
    }

    return mapping.get(name.lower(), name)


# ============================================================
# COLLECT RESULTS
# ============================================================

records = []

# ONLY search inside ports directories
metrics_files = list(RUNS_DIR.glob("**/ports/seed*/METRICS_ROW.json"))

print("=" * 70)
print("Per-Attack PORTS Aggregation")
print("=" * 70)

print(f"Root : {ROOT}")
print(f"Found METRICS_ROW files: {len(metrics_files)}")
print()


for metrics_path in sorted(metrics_files):

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    except Exception as e:
        print(f"[WARNING] Could not read: {metrics_path}")
        print(f"          {e}")
        continue

    # Safety check: only ports
    if "ports" not in metrics_path.parts:
        continue

    model = get_model_name(metrics_path)
    seed = get_seed(metrics_path)

    # Make sure this really is a ports run
    if metrics.get("feature_set") != "ports":
        print(
            f"[WARNING] Skipping non-ports file: "
            f"{metrics_path}"
        )
        continue

    # --------------------------------------------------------
    # Extract only per-attack results
    # --------------------------------------------------------

    for key, value in metrics.items():

        if key in ATTACKS:
            attack = normalize_attack_name(key)

            try:
                value = float(value)
            except (TypeError, ValueError):
                print(
                    f"[WARNING] Invalid value for "
                    f"{model}/{attack}/seed{seed}: {value}"
                )
                continue

            records.append(
                {
                    "Model": model,
                    "Attack": attack,
                    "Seed": seed,
                    "Detection_Rate": value,
                    "Source_File": str(metrics_path),
                }
            )


# ============================================================
# CHECK DATA
# ============================================================

if not records:
    raise RuntimeError(
        "No per-attack ports results were found."
    )


raw_df = pd.DataFrame(records)

print("Models found:")
for model in sorted(raw_df["Model"].unique()):
    count = (raw_df["Model"] == model).sum()
    seeds = sorted(
        raw_df.loc[
            raw_df["Model"] == model,
            "Seed"
        ].dropna().unique()
    )

    print(
        f"  {model:<10} "
        f"{count:>3} attack records | "
        f"seeds={seeds}"
    )

print()


# ============================================================
# AGGREGATE ACROSS SEEDS
# ============================================================

aggregated = (
    raw_df
    .groupby(["Model", "Attack"])["Detection_Rate"]
    .agg(
        N="count",
        Mean="mean",
        Std="std",
        Min="min",
        Max="max",
    )
    .reset_index()
)


# If only one seed exists, pandas gives NaN for Std.
aggregated["Std"] = aggregated["Std"].fillna(0.0)


# ============================================================
# ROUNDING
# ============================================================

aggregated["Mean"] = aggregated["Mean"].round(4)
aggregated["Std"] = aggregated["Std"].round(4)
aggregated["Min"] = aggregated["Min"].round(4)
aggregated["Max"] = aggregated["Max"].round(4)


# ============================================================
# ORDERING
# ============================================================

model_order = [
    # DRL
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B9",
    "B10",
    "B11",

    # Supervised
    "RF",
    "XGB",
    "MLP",
    "LSTM",
]

attack_order = [
    "Backdoor",
    "Brute Force",
    "Shellcode",
    "Theft",
    "ransomware",
]

aggregated["Model"] = pd.Categorical(
    aggregated["Model"],
    categories=model_order,
    ordered=True,
)

aggregated["Attack"] = pd.Categorical(
    aggregated["Attack"],
    categories=attack_order,
    ordered=True,
)

aggregated = (
    aggregated
    .sort_values(["Model", "Attack"])
    .reset_index(drop=True)
)


# ============================================================
# SAVE ONE FILE
# ============================================================

aggregated.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("=" * 70)
print("DONE")
print("=" * 70)
print(f"Output: {OUTPUT_FILE}")
print()
print(aggregated.to_string(index=False))
print()