"""Experiment catalog: B0–B10 (paper ablation), B11 (exploratory extension), and advanced algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.data.constants import EXPERIMENTS_DIR, PROJECT_ROOT
from agent.training.advanced.registry import (
    ADVANCED_ORDER,
    ADVANCED_REGISTRY,
    list_advanced,
)

EXPERIMENT_ORDER = [
    "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11",
]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    title: str
    config_path: Path
    family_dir: Path
    compatibility_csv: Path | None = None
    description: str = ""
    group: str = "baseline"  # baseline | advanced


def _spec(
    experiment_id: str,
    title: str,
    folder: str,
    description: str,
    *,
    config_path: Path | None = None,
    group: str = "baseline",
) -> ExperimentSpec:
    # One clean report per model (written by layout.save_standard_run)
    report = EXPERIMENTS_DIR / f"{experiment_id}_zero_day_per_attack_metrics.csv"
    return ExperimentSpec(
        experiment_id=experiment_id,
        title=title,
        config_path=config_path or (PROJECT_ROOT / "configs" / f"{experiment_id}.yaml"),
        family_dir=EXPERIMENTS_DIR / folder,
        compatibility_csv=report,
        description=description,
        group=group,
    )


CATALOG: dict[str, ExperimentSpec] = {
    "B0": _spec("B0", "Original Stacked-LSTM DQN", "B0_original_baseline", "Paper-style softmax DQN baseline."),
    "B1": _spec("B1", "Double DQN + target network", "B1_double_dqn", "Linear-Q Double DQN."),
    "B2": _spec("B2", "Episode-based epsilon decay", "B2_episode_epsilon", "B1 + episode-level epsilon decay."),
    "B3": _spec("B3", "Larger replay buffer", "B3_large_replay", "B2 + replay capacity 10_000."),
    "B4": _spec("B4", "Prioritized Experience Replay", "B4_per", "Double DQN + PER."),
    "B5": _spec("B5", "Cost-sensitive reward + PER", "B5_cost_sensitive", "Cost-sensitive IDS rewards with PER."),
    "B6": _spec("B6", "Dueling Double DQN + PER", "B6_dueling_per", "Dueling Q-head + PER."),
    "B7": _spec("B7", "Stratified attack-family sampling", "B7_stratified", "Family-stratified episode sampling + PER."),
    "B8": _spec("B8", "Hybrid stratified sampling", "B8_hybrid_sampling", "Natural + stratified family sampling + PER."),
    "B9": _spec("B9", "Adaptive hybrid sampling", "B9_adaptive_hybrid", "Adaptive family quotas from EMA accuracy + PER."),
    "B10": _spec("B10", "Restrained adaptive hybrid sampling", "B10_restrained_adaptive", "Adaptive family quotas with min/max caps + PER."),
    "B11": _spec("B11", "Family-aware sampling + investigate action + post-alert queue effect", "B11_investigate", "B10 family-aware sampling + investigate action + post-alert queue effect + cost-sensitive reward."),
}

# Advanced algorithms (advance-AI) — registered, implementations land incrementally.
for _aid in ADVANCED_ORDER:
    _adv = ADVANCED_REGISTRY[_aid]
    CATALOG[_aid] = _spec(
        _adv.algorithm_id,
        _adv.title,
        _adv.family_dir.name,
        f"[{_adv.status}] {_adv.description}",
        config_path=_adv.config_path,
        group="advanced",
    )


def list_experiments(*, group: str | None = None) -> list[str]:
    order = EXPERIMENT_ORDER + ADVANCED_ORDER
    ids = [eid for eid in order if eid in CATALOG]
    if group is None:
        return ids
    return [eid for eid in ids if CATALOG[eid].group == group]


def list_baseline_experiments() -> list[str]:
    return list_experiments(group="baseline")


def list_advanced_experiments() -> list[str]:
    return list_advanced()


def get_experiment(experiment_id: str) -> ExperimentSpec:
    key = experiment_id.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "QR-DQN": "QR_DQN",
        "QRDQN": "QR_DQN",
        "M-DQN": "M_DQN",
        "MDQN": "M_DQN",
        "MUNCHAUSEN": "M_DQN",
        "APE-X": "APEX",
        "APE_X": "APEX",
        "A3C": "A2C",
        "RAINBOW_DQN": "RAINBOW",
    }
    key = aliases.get(key, key)
    if key not in CATALOG:
        known = ", ".join(list_experiments())
        raise KeyError(f"Unknown experiment '{experiment_id}'. Known: {known}")
    return CATALOG[key]
