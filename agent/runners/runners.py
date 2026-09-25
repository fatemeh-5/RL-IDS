"""High-level rebuild runners for catalog experiments B0–B10."""

from __future__ import annotations

from typing import Any

from pathlib import Path

from agent.utils.config import load_config
from agent.data.constants import feature_set_cache_dir, feature_set_columns
from agent.data.pipeline import PreparedData, load_or_build_cache
from agent.envs import (
    BaselineIDSEnvironment,
    CostSensitiveIDSEnvironment,
    CustomSamplingIDSEnvironment,
    FamilyAwareIDSEnvironment,
    InvestigateIDSEnvironment,
)
from agent.evaluation.metrics import evaluate_all
from agent.runners.catalog import get_experiment, list_baseline_experiments, list_experiments
from agent.runners.layout import create_run_dirs, save_standard_run
from agent.training.double_dqn import train_loop
from agent.training.baseline import train_baseline
from agent.training.per import train_per
from agent.training.advanced.dispatch import TRAINERS, train_advanced


def _require_balanced(data: PreparedData) -> None:
    if not data.has_balanced:
        raise RuntimeError(
            "Balanced training cache missing. Run:\n"
            "  python scripts/prepare_data.py --force"
        )


def _build_env(cfg: dict[str, Any], data: PreparedData):
    env_kind = str(cfg.get("env", "baseline")).lower()
    episode_length = int(cfg.get("episode_length", 256))
    seed = int(cfg.get("seed", 42))

    if env_kind == "custom":
        _require_balanced(data)
        return CustomSamplingIDSEnvironment(
            features=data.X_train_balanced,
            labels=data.y_train_balanced,
            benign_ratio=float(cfg.get("benign_ratio", 0.5)),
            episode_length=episode_length,
            random_state=seed,
        )

    if env_kind == "cost_sensitive":
        _require_balanced(data)
        return CostSensitiveIDSEnvironment(
            features=data.X_train_balanced,
            labels=data.y_train_balanced,
            episode_length=episode_length,
            random_state=seed,
            reward_tp=float(cfg.get("reward_tp", 5.0)),
            reward_tn=float(cfg.get("reward_tn", 2.0)),
            reward_fp=float(cfg.get("reward_fp", -3.0)),
            reward_fn=float(cfg.get("reward_fn", -10.0)),
        )

    def _family_kwargs(features, labels, attack_names) -> dict[str, Any]:
        return dict(
            features=features,
            labels=labels,
            attack_names=attack_names,
            episode_length=episode_length,
            benign_per_episode=int(cfg.get("benign_per_episode", 128)),
            natural_attack_per_episode=int(cfg.get("natural_attack_per_episode", 64)),
            stratified_attack_per_episode=int(
                cfg.get("stratified_attack_per_episode", 64)
            ),
            adaptive_mix=float(cfg.get("adaptive_mix", 0.70)),
            ema_decay=float(cfg.get("ema_decay", 0.80)),
            min_family_quota=int(cfg.get("min_family_quota", 1)),
            max_family_quota=(
                None
                if cfg.get("max_family_quota", None) is None
                else int(cfg.get("max_family_quota"))
            ),
            random_state=seed,
        )

    if env_kind == "family":
        # Family sampling uses original (pre-SMOTE) train rows + attack names.
        return FamilyAwareIDSEnvironment(
            mode=str(cfg.get("family_mode", "hybrid")),
            **_family_kwargs(data.X_train_scaled, data.y_train, data.train_attack),
        )

    if env_kind == "investigate":
        # B11: family-aware sampling + investigate action + post-alert queue effect.
        # Same natural (pre-SMOTE) train rows + attack names as the family env.
        return InvestigateIDSEnvironment(
            mode=str(cfg.get("family_mode", "restrained")),
            investigate_budget=int(cfg.get("investigate_budget", 20)),
            investigate_penalty_exhausted=float(
                cfg.get("investigate_penalty_exhausted", -2.0)
            ),
            post_alert_countdown_length=int(cfg.get("post_alert_countdown_length", 5)),
            post_alert_benign_boost=float(cfg.get("post_alert_benign_boost", 0.30)),
            reward_tp=float(cfg.get("reward_tp", 5.0)),
            reward_tn=float(cfg.get("reward_tn", 2.0)),
            reward_fp=float(cfg.get("reward_fp", -3.0)),
            reward_fn=float(cfg.get("reward_fn", -10.0)),
            **_family_kwargs(data.X_train_scaled, data.y_train, data.train_attack),
        )



    _require_balanced(data)
    return BaselineIDSEnvironment(
        features=data.X_train_balanced,
        labels=data.y_train_balanced,
        episode_length=episode_length,
        random_state=seed,
    )


def _train(cfg: dict[str, Any], env, checkpoint_dir, mode: str) -> dict[str, Any]:
    trainer = str(cfg.get("trainer", "double")).lower()
    common = dict(
        episodes=int(cfg.get("episodes", 31)),
        episode_length=int(cfg.get("episode_length", 256)),
        batch_size=int(cfg.get("batch_size", 16)),
        gamma=float(cfg.get("gamma", 0.97)),
        learning_rate=float(cfg.get("learning_rate", 0.001)),
        seed=int(cfg.get("seed", 42)),
        mode=mode,
        checkpoint_dir=checkpoint_dir,
        save_every_episodes=int(cfg.get("save_every_episodes", 5)),
    )

    if trainer == "baseline":
        return train_baseline(
            env,
            replay_capacity=int(cfg.get("replay_capacity", 1000)),
            epsilon_start=float(cfg.get("epsilon_start", 1.0)),
            epsilon_min=float(cfg.get("epsilon_min", 0.05)),
            epsilon_decay=float(cfg.get("epsilon_decay", 0.996)),
            **common,
        )

    if trainer == "per":
        return train_per(
            env,
            model_type=str(cfg.get("model_type", "double")),
            replay_capacity=int(cfg.get("replay_capacity", 1000)),
            per_alpha=float(cfg.get("per_alpha", 0.60)),
            beta_start=float(cfg.get("beta_start", 0.40)),
            beta_end=float(cfg.get("beta_end", 1.00)),
            priority_epsilon=float(cfg.get("priority_epsilon", 1e-6)),
            target_update_interval=int(cfg.get("target_update_interval", 250)),
            epsilon_start=float(cfg.get("epsilon_start", 1.0)),
            epsilon_min=float(cfg.get("epsilon_min", 0.05)),
            epsilon_decay=float(cfg.get("epsilon_decay", 0.90)),
            **common,
        )

    if trainer in TRAINERS:
        # Advanced algorithms (PPO, Rainbow, C51, …) — pass full cfg for algo-specific knobs.
        return train_advanced(trainer, env, cfg=cfg, **common)

    # default: double DQN uniform replay (B1/B2/B3)
    return train_loop(
        env,
        replay_capacity=int(cfg.get("replay_capacity", 1000)),
        target_update_interval=int(cfg.get("target_update_interval", 250)),
        epsilon_start=float(cfg.get("epsilon_start", 1.0)),
        epsilon_min=float(cfg.get("epsilon_min", 0.05)),
        epsilon_decay=float(cfg.get("epsilon_decay", 0.90)),
        epsilon_schedule=str(cfg.get("epsilon_schedule", "episode")),
        **common,
    )


def describe_trainer(experiment_id: str) -> dict[str, str]:
    """Resolve the trainer / model-builder / env that `rebuild_experiment` will
    actually use for `experiment_id`, straight from its configs/<ID>.yaml.

    Mirrors the branching in `_train`/`_build_env` exactly (same `trainer`,
    `model_type`, `env` keys, same defaults) so this can never drift from what
    training actually does — used as a preflight fidelity check by the
    multi-seed sweep so a shared/mis-resolved trainer can't silently train the
    wrong algorithm for a config.
    """
    spec = get_experiment(experiment_id)
    cfg = load_config(spec.config_path)

    trainer = str(cfg.get("trainer", "double")).lower()
    if trainer == "baseline":
        model_builder = "build_baseline_dqn"
    elif trainer == "per":
        model_type = str(cfg.get("model_type", "double")).lower()
        model_builder = "build_dueling_dqn" if model_type == "dueling" else "build_double_dqn"
    elif trainer in TRAINERS:
        model_builder = f"advanced:{trainer}"
    else:
        # _train's fallback branch (B1-B3): uniform-replay double DQN.
        model_builder = "build_double_dqn"
        trainer = "double"

    env_kind = str(cfg.get("env", "baseline")).lower()
    n_actions = 3 if env_kind == "investigate" else 2

    return {
        "config": experiment_id,
        "trainer": trainer,
        "model_builder": f"{model_builder}(n_actions={n_actions})",
        "env": env_kind,
    }


def rebuild_experiment(
    experiment_id: str,
    *,
    mode: str | None = None,
    episodes_override: int | None = None,
    feature_set: str = "ports",
    seed_override: int | None = None,
    run_dir: Path | None = None,
    mirror_family: bool = True,
    publish_top_level_csv: bool | None = None,
) -> dict[str, Any]:
    """Train/eval one catalog experiment and write Untitled-compatible artifacts.

    feature_set: "ports" (default, original 10 columns) or "noports" (drops
        L4_SRC_PORT/L4_DST_PORT) — see agent.data.constants.FEATURE_SETS.
    seed_override: override the config's training seed (env sampling, weight
        init, replay/exploration randomness). Does not affect the data split.
    run_dir: write run artifacts directly here instead of the family_dir /
        timestamp scheme (used by the multi-seed sweep so runs don't collide).
    mirror_family / publish_top_level_csv: when False, skip writing/overwriting
        the shared "canonical" per-experiment files in experiments/<family>/
        and experiments/<ID>_zero_day_per_attack_metrics.csv. Sweep callers
        should pass False for both so repeated seeds don't clobber them.
    """

    spec = get_experiment(experiment_id)
    if not spec.config_path.exists():
        raise FileNotFoundError(f"Missing config: {spec.config_path}")

    cfg = load_config(spec.config_path)
    yaml_episodes = int(cfg.get("episodes", 31))
    mode = mode or cfg.get("mode", "train")
    if episodes_override is not None:
        cfg["episodes"] = int(episodes_override)
    if seed_override is not None:
        cfg["seed"] = int(seed_override)
    if publish_top_level_csv is None:
        publish_top_level_csv = mirror_family

    print("=" * 72)
    print(f"Rebuild {spec.experiment_id}: {spec.title}")
    print(f"Mode: {mode}  Feature set: {feature_set}  Seed: {cfg.get('seed')}")
    print(f"Config: {spec.config_path}")
    print("=" * 72)

    data = load_or_build_cache(
        cache_dir=feature_set_cache_dir(feature_set),
        require_balanced=True,
        feature_columns=feature_set_columns(feature_set),
    )
    dirs = create_run_dirs(spec, run_dir=run_dir)
    env = _build_env(cfg, data)
    result = _train(cfg, env, dirs["checkpoints_dir"], mode)
    model_filename = (
        "baseline_model.keras"
        if str(cfg.get("trainer", "")).lower() == "baseline"
        else "online_model.keras"
    )

    print("Evaluating ...")
    eval_result = evaluate_all(result["online_model"], data, verbose=0)
    print(eval_result["metrics_df"].round(4).to_string(index=False))
    print("\nZero-day:")
    print(eval_result["zero_day_metrics"])
    print(
        eval_result["per_attack_df"][
            ["Attack", "Samples", "Detection Rate (%)", "Miss Rate (%)"]
        ]
        .round(2)
        .to_string(index=False)
    )

    configuration = {
        "experiment_id": spec.experiment_id,
        "run_id": dirs["run_id"],
        "title": spec.title,
        "description": spec.description,
        "feature_set": feature_set,
        **{k: v for k, v in cfg.items() if not str(k).startswith("_")},
    }
    publish_top = (
        int(cfg.get("episodes", yaml_episodes)) >= yaml_episodes
    ) and publish_top_level_csv

    save_standard_run(
        spec=spec,
        dirs=dirs,
        configuration=configuration,
        online_model=result["online_model"],
        target_model=result.get("target_model"),
        history_df=result.get("history_df"),
        eval_result=eval_result,
        data=data,
        model_filename=model_filename,
        publish_top_level_csv=publish_top,
        mirror_family=mirror_family,
    )

    return {
        "spec": spec,
        "dirs": dirs,
        "result": result,
        "eval_result": eval_result,
    }


def rebuild_many(
    experiment_ids: list[str] | None = None,
    *,
    mode: str | None = None,
    episodes_override: int | None = None,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Rebuild multiple experiments in catalog order and save each run."""

    ids = experiment_ids or list_baseline_experiments()
    summaries: list[dict[str, Any]] = []
    for eid in ids:
        try:
            out = rebuild_experiment(
                eid, mode=mode, episodes_override=episodes_override
            )
            summaries.append(
                {
                    "experiment_id": eid,
                    "status": "ok",
                    "run_dir": str(out["dirs"]["run_dir"]),
                }
            )
        except Exception as exc:  # noqa: BLE001 - batch runner should continue
            print(f"[ERROR] {eid}: {exc}")
            summaries.append({"experiment_id": eid, "status": "error", "error": str(exc)})
            if not continue_on_error:
                raise
    return summaries
