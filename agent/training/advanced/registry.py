"""Catalog of advanced RL algorithms planned for the advance-AI branch.

IDS note: the env is Discrete(2) (Benign/Attack). Value-based and discrete
policy-gradient methods map directly. Continuous-control methods (SAC/TD3/DDPG)
need a discrete or hybrid adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.data.constants import EXPERIMENTS_DIR, PROJECT_ROOT


@dataclass(frozen=True)
class AdvancedSpec:
    algorithm_id: str
    title: str
    family: str  # value | policy | actor_critic | distributed
    action_space: str  # discrete | continuous_adapt
    trainer: str
    status: str  # planned | wip | ready
    description: str
    config_path: Path
    family_dir: Path

    @property
    def experiment_id(self) -> str:
        return self.algorithm_id


def _spec(
    algorithm_id: str,
    title: str,
    family: str,
    action_space: str,
    trainer: str,
    folder: str,
    description: str,
    status: str = "ready",
) -> AdvancedSpec:
    return AdvancedSpec(
        algorithm_id=algorithm_id,
        title=title,
        family=family,
        action_space=action_space,
        trainer=trainer,
        status=status,
        description=description,
        config_path=PROJECT_ROOT / "configs" / "advanced" / f"{algorithm_id}.yaml",
        family_dir=EXPERIMENTS_DIR / folder,
    )


# Recommended build order for binary IDS (Discrete-2):
#   1) PPO  2) C51  3) QR-DQN  4) Rainbow  5) M-DQN  6) A2C
#   7) REINFORCE  8) TRPO  9) R2D2  10) Ape-X  11) IMPALA
#   12) ACER  13) ACKTR  14) SAC-d  15) TD3-d  16) DDPG-d
ADVANCED_ORDER: list[str] = [
    "PPO",
    "C51",
    "QR_DQN",
    "RAINBOW",
    "M_DQN",
    "A2C",
    "REINFORCE",
    "TRPO",
    "R2D2",
    "APEX",
    "IMPALA",
    "ACER",
    "ACKTR",
    "SAC",
    "TD3",
    "DDPG",
]

ADVANCED_REGISTRY: dict[str, AdvancedSpec] = {
    "PPO": _spec(
        "PPO",
        "Proximal Policy Optimization",
        "actor_critic",
        "discrete",
        "ppo",
        "ADV_PPO",
        "Clipped surrogate policy gradient; strong discrete baseline.",
    ),
    "C51": _spec(
        "C51",
        "Categorical DQN (C51)",
        "value",
        "discrete",
        "c51",
        "ADV_C51",
        "Distributional RL over a fixed categorical support.",
    ),
    "QR_DQN": _spec(
        "QR_DQN",
        "Quantile Regression DQN",
        "value",
        "discrete",
        "qr_dqn",
        "ADV_QR_DQN",
        "Learns a quantile distribution of returns.",
    ),
    "RAINBOW": _spec(
        "RAINBOW",
        "Rainbow DQN",
        "value",
        "discrete",
        "rainbow",
        "ADV_RAINBOW",
        "Combines Double, Dueling, PER, N-step, Distributional, Noisy nets.",
    ),
    "M_DQN": _spec(
        "M_DQN",
        "Munchausen DQN",
        "value",
        "discrete",
        "m_dqn",
        "ADV_M_DQN",
        "Adds a scaled log-policy bonus to the DQN target (Munchausen).",
    ),
    "A2C": _spec(
        "A2C",
        "Advantage Actor-Critic (A2C / A3C-sync)",
        "actor_critic",
        "discrete",
        "a2c",
        "ADV_A2C",
        "Synchronous A2C; A3C is the async multi-worker variant.",
    ),
    "REINFORCE": _spec(
        "REINFORCE",
        "REINFORCE (vanilla policy gradient)",
        "policy",
        "discrete",
        "reinforce",
        "ADV_REINFORCE",
        "Monte-Carlo policy gradient with optional baseline.",
    ),
    "TRPO": _spec(
        "TRPO",
        "Trust Region Policy Optimization",
        "actor_critic",
        "discrete",
        "trpo",
        "ADV_TRPO",
        "KL-constrained natural policy gradient (harder than PPO).",
    ),
    "R2D2": _spec(
        "R2D2",
        "Recurrent Replay Distributed DQN",
        "value",
        "discrete",
        "r2d2",
        "ADV_R2D2",
        "LSTM + stored recurrent state + burn-in + prioritized sequence replay.",
    ),
    "APEX": _spec(
        "APEX",
        "Ape-X DQN",
        "distributed",
        "discrete",
        "apex",
        "ADV_APEX",
        "Distributed actors + prioritized replay learner (Ape-X).",
    ),
    "IMPALA": _spec(
        "IMPALA",
        "Importance-Weighted Actor-Learner",
        "distributed",
        "discrete",
        "impala",
        "ADV_IMPALA",
        "V-trace off-policy actor-learner for scalable training.",
    ),
    "ACER": _spec(
        "ACER",
        "Actor-Critic with Experience Replay",
        "actor_critic",
        "discrete",
        "acer",
        "ADV_ACER",
        "Off-policy actor-critic with truncated IS and trust region.",
    ),
    "ACKTR": _spec(
        "ACKTR",
        "Actor-Critic using Kronecker-Factored Trust Region",
        "actor_critic",
        "discrete",
        "acktr",
        "ADV_ACKTR",
        "K-FAC natural gradient actor-critic.",
    ),
    "SAC": _spec(
        "SAC",
        "Soft Actor-Critic (discrete adaptation)",
        "actor_critic",
        "continuous_adapt",
        "sac",
        "ADV_SAC",
        "Discrete SAC (max-entropy) for Binary IDS actions.",
    ),
    "TD3": _spec(
        "TD3",
        "Twin Delayed DDPG (discrete adaptation)",
        "actor_critic",
        "continuous_adapt",
        "td3",
        "ADV_TD3",
        "Requires discrete/hybrid action mapping for IDS.",
    ),
    "DDPG": _spec(
        "DDPG",
        "Deep Deterministic Policy Gradient (discrete adaptation)",
        "actor_critic",
        "continuous_adapt",
        "ddpg",
        "ADV_DDPG",
        "Requires discrete/hybrid action mapping for IDS.",
    ),
}


def list_advanced() -> list[str]:
    return [aid for aid in ADVANCED_ORDER if aid in ADVANCED_REGISTRY]


def get_advanced(algorithm_id: str) -> AdvancedSpec:
    key = algorithm_id.strip().upper().replace("-", "_").replace(" ", "_")
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
    if key not in ADVANCED_REGISTRY:
        known = ", ".join(list_advanced())
        raise KeyError(f"Unknown advanced algorithm '{algorithm_id}'. Known: {known}")
    return ADVANCED_REGISTRY[key]
