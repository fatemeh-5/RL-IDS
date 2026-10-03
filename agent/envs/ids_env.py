"""IDS Gym environments."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BaselineIDSEnvironment(gym.Env):
    """
    Binary IDS environment with uniform episode sampling.

    Customize sampling by subclassing and overriding `_build_episode_indices`.
    That is the recommended hook for new research algorithms.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        episode_length: int = 256,
        random_state: int = 42,
    ):
        super().__init__()

        if len(features) != len(labels):
            raise ValueError("features and labels length mismatch.")
        if episode_length < 2:
            raise ValueError("episode_length must be at least 2.")
        if episode_length > len(features):
            raise ValueError("episode_length cannot exceed dataset size.")

        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int8)
        self.episode_length = int(episode_length)
        self.rng = np.random.default_rng(random_state)

        self.n_samples = len(self.features)
        self.n_features = self.features.shape[1]

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )

        self.episode_indices = None
        self.current_step = 0

    def _build_episode_indices(self) -> np.ndarray:
        """Override this method to implement a new sampling algorithm."""
        return self.rng.choice(
            self.n_samples,
            size=self.episode_length,
            replace=False,
        ).astype(np.int64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_indices = self._build_episode_indices()
        self.current_step = 0
        first_index = int(self.episode_indices[self.current_step])
        state = self.features[first_index]
        info = {
            "sample_index": first_index,
            "true_label": int(self.labels[first_index]),
        }
        return state, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        sample_index = int(self.episode_indices[self.current_step])
        true_label = int(self.labels[sample_index])
        reward = 1.0 if action == true_label else -1.0

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        if terminated:
            next_state = np.zeros(self.n_features, dtype=np.float32)
        else:
            next_index = int(self.episode_indices[self.current_step])
            next_state = self.features[next_index]

        info = {
            "sample_index": sample_index,
            "true_label": true_label,
            "action": int(action),
            "correct": bool(action == true_label),
        }
        return next_state, reward, terminated, truncated, info


class CustomSamplingIDSEnvironment(BaselineIDSEnvironment):
    """
    Example extension point for a new algorithm.

    Replace `_build_episode_indices` with your research sampling logic.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        attack_names: np.ndarray | None = None,
        benign_ratio: float = 0.5,
        episode_length: int = 256,
        random_state: int = 42,
    ):
        super().__init__(
            features=features,
            labels=labels,
            episode_length=episode_length,
            random_state=random_state,
        )
        self.attack_names = (
            None if attack_names is None else np.asarray(attack_names, dtype=object)
        )
        self.benign_ratio = float(benign_ratio)
        self.benign_indices = np.where(self.labels == 0)[0]
        self.attack_indices = np.where(self.labels == 1)[0]

    def _build_episode_indices(self) -> np.ndarray:
        n_benign = int(round(self.episode_length * self.benign_ratio))
        n_attack = self.episode_length - n_benign

        def _sample(pool: np.ndarray, size: int) -> np.ndarray:
            replace = len(pool) < size
            return self.rng.choice(pool, size=size, replace=replace).astype(np.int64)

        indices = np.concatenate(
            [
                _sample(self.benign_indices, n_benign),
                _sample(self.attack_indices, n_attack),
            ]
        )
        self.rng.shuffle(indices)
        return indices


def calculate_cost_sensitive_reward(
    true_label: int,
    action: int,
    *,
    tp: float = 5.0,
    tn: float = 2.0,
    fp: float = -3.0,
    fn: float = -10.0,
) -> tuple[float, str]:
    true_label = int(true_label)
    action = int(action)
    if true_label == 1 and action == 1:
        return tp, "True Positive"
    if true_label == 0 and action == 0:
        return tn, "True Negative"
    if true_label == 0 and action == 1:
        return fp, "False Positive"
    return fn, "False Negative"


class CostSensitiveIDSEnvironment(BaselineIDSEnvironment):
    """B5 cost-sensitive reward matrix on uniform episode sampling."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        episode_length: int = 256,
        random_state: int = 42,
        reward_tp: float = 5.0,
        reward_tn: float = 2.0,
        reward_fp: float = -3.0,
        reward_fn: float = -10.0,
    ):
        super().__init__(
            features=features,
            labels=labels,
            episode_length=episode_length,
            random_state=random_state,
        )
        self.reward_tp = float(reward_tp)
        self.reward_tn = float(reward_tn)
        self.reward_fp = float(reward_fp)
        self.reward_fn = float(reward_fn)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        sample_index = int(self.episode_indices[self.current_step])
        true_label = int(self.labels[sample_index])
        reward, outcome = calculate_cost_sensitive_reward(
            true_label,
            action,
            tp=self.reward_tp,
            tn=self.reward_tn,
            fp=self.reward_fp,
            fn=self.reward_fn,
        )

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False
        if terminated:
            next_state = np.zeros(self.n_features, dtype=np.float32)
        else:
            next_index = int(self.episode_indices[self.current_step])
            next_state = self.features[next_index]

        info = {
            "sample_index": sample_index,
            "true_label": true_label,
            "action": int(action),
            "correct": bool(action == true_label),
            "outcome": outcome,
        }
        return next_state, reward, terminated, truncated, info


class FamilyAwareIDSEnvironment(BaselineIDSEnvironment):
    """
    Family-aware episode sampling for B7–B10 style experiments.

    Uses original (pre-SMOTE) train features + attack family names.
    Modes:
      - stratified: equal-ish quotas across known attack families
      - hybrid: benign + natural attacks + stratified attacks
      - adaptive / restrained: boost harder families via EMA accuracy
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        attack_names: np.ndarray,
        *,
        mode: str = "hybrid",
        episode_length: int = 256,
        benign_per_episode: int = 128,
        natural_attack_per_episode: int = 64,
        stratified_attack_per_episode: int = 64,
        adaptive_mix: float = 0.70,
        ema_decay: float = 0.80,
        min_family_quota: int = 1,
        max_family_quota: int | None = None,
        random_state: int = 42,
    ):
        super().__init__(
            features=features,
            labels=labels,
            episode_length=episode_length,
            random_state=random_state,
        )
        self.attack_names = np.asarray(attack_names, dtype=object)
        if len(self.attack_names) != len(self.labels):
            raise ValueError("attack_names must align with labels.")

        self.mode = mode
        self.benign_per_episode = int(benign_per_episode)
        self.natural_attack_per_episode = int(natural_attack_per_episode)
        self.stratified_attack_per_episode = int(stratified_attack_per_episode)
        self.adaptive_mix = float(adaptive_mix)
        self.ema_decay = float(ema_decay)
        self.min_family_quota = int(min_family_quota)
        self.max_family_quota = max_family_quota
        self.episode_number = 0

        self.benign_indices = np.where(self.labels == 0)[0]
        self.attack_indices = np.where(self.labels == 1)[0]
        families = sorted(
            {
                str(name)
                for name in self.attack_names[self.attack_indices]
                if str(name).lower() != "benign"
            }
        )
        self.families = families
        self.family_pools = {
            family: np.where(
                (self.labels == 1) & (self.attack_names == family)
            )[0]
            for family in families
        }
        self.family_ema_accuracy = {family: 0.5 for family in families}
        self._last_episode_family_correct: dict[str, list[int]] = {
            family: [] for family in families
        }

        expected = (
            self.benign_per_episode
            + self.natural_attack_per_episode
            + self.stratified_attack_per_episode
        )
        if expected != self.episode_length:
            raise ValueError(
                f"Episode composition {expected} != episode_length {self.episode_length}"
            )

    def _sample_pool(self, pool: np.ndarray, size: int) -> np.ndarray:
        if size <= 0:
            return np.zeros(0, dtype=np.int64)
        if len(pool) == 0:
            # fallback to any attack
            pool = self.attack_indices
        replace = len(pool) < size
        return self.rng.choice(pool, size=size, replace=replace).astype(np.int64)

    def _family_quotas(self, total: int) -> dict[str, int]:
        n = len(self.families)
        if n == 0 or total <= 0:
            return {}

        if self.mode in {"adaptive", "restrained"}:
            difficulties = []
            for family in self.families:
                acc = self.family_ema_accuracy[family]
                difficulties.append(max(1e-6, 1.0 - acc))
            difficulties = np.asarray(difficulties, dtype=np.float64)
            uniform = np.full(n, 1.0 / n)
            adaptive = difficulties / difficulties.sum()
            probs = (
                (1.0 - self.adaptive_mix) * uniform
                + self.adaptive_mix * adaptive
            )
            probs = probs / probs.sum()
            return self._allocate_quotas(probs, total)

        # stratified / hybrid: rotate equal quotas
        base = total // n
        rem = total - base * n
        quotas = {family: base for family in self.families}
        start = self.episode_number % n
        for offset in range(rem):
            family = self.families[(start + offset) % n]
            quotas[family] += 1
        return quotas

    def _allocate_quotas(self, probs: np.ndarray, total: int) -> dict[str, int]:
        """Min quota per family, then split the rest proportionally to `probs`.

        Largest-remainder allocation, capped at max_family_quota; capacity freed
        by capped families is re-split among the open ones. If every family
        hits the cap, the shortfall is left for `_build_episode_indices` to pad.
        """
        n = len(self.families)
        base = self.min_family_quota
        if base * n > total:
            base = total // n
        quotas = np.full(n, base, dtype=np.int64)
        cap = total if self.max_family_quota is None else int(self.max_family_quota)
        remaining = total - base * n

        while remaining > 0:
            room = np.maximum(cap - quotas, 0)
            open_mask = room > 0
            if not open_mask.any():
                break
            weights = np.where(open_mask, probs, 0.0)
            share = weights / weights.sum() * remaining
            add = np.minimum(np.floor(share).astype(np.int64), room)
            if add.sum() == 0:
                # Every share < 1: hand out single units by largest share.
                for idx in np.argsort(-share, kind="stable"):
                    if remaining <= 0:
                        break
                    if room[idx] > 0:
                        quotas[idx] += 1
                        remaining -= 1
                break
            quotas += add
            remaining -= int(add.sum())

        return {family: int(q) for family, q in zip(self.families, quotas)}

    def _sample_family_quota(self, family: str, quota: int) -> np.ndarray:
        """Override to change how a single family's quota is filled."""
        return self._sample_pool(self.family_pools[family], quota)

    def _build_episode_indices(self) -> np.ndarray:
        benign = self._sample_pool(self.benign_indices, self.benign_per_episode)

        if self.mode == "stratified":
            natural = np.zeros(0, dtype=np.int64)
            strat_budget = (
                self.natural_attack_per_episode + self.stratified_attack_per_episode
            )
        else:
            natural = self._sample_pool(
                self.attack_indices, self.natural_attack_per_episode
            )
            strat_budget = self.stratified_attack_per_episode

        quotas = self._family_quotas(strat_budget)
        stratified_parts = [
            self._sample_family_quota(family, quota)
            for family, quota in quotas.items()
            if quota > 0
        ]
        stratified = (
            np.concatenate(stratified_parts)
            if stratified_parts
            else np.zeros(0, dtype=np.int64)
        )

        indices = np.concatenate([benign, natural, stratified])
        if len(indices) != self.episode_length:
            # pad/truncate safely
            if len(indices) < self.episode_length:
                pad = self._sample_pool(
                    self.attack_indices, self.episode_length - len(indices)
                )
                indices = np.concatenate([indices, pad])
            indices = indices[: self.episode_length]

        self.rng.shuffle(indices)
        self._episode_family_tracker = [
            str(self.attack_names[i]) if self.labels[i] == 1 else "Benign"
            for i in indices
        ]
        return indices.astype(np.int64)

    def reset(self, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        self._last_episode_family_correct = {family: [] for family in self.families}
        return state, info

    def step(self, action):
        sample_index = int(self.episode_indices[self.current_step])
        true_label = int(self.labels[sample_index])
        family = str(self.attack_names[sample_index])
        next_state, reward, terminated, truncated, info = BaselineIDSEnvironment.step(
            self, action
        )
        if true_label == 1 and family in self._last_episode_family_correct:
            self._last_episode_family_correct[family].append(int(info["correct"]))
        if terminated:
            for fam, outcomes in self._last_episode_family_correct.items():
                if not outcomes:
                    continue
                ep_acc = float(np.mean(outcomes))
                prev = self.family_ema_accuracy[fam]
                self.family_ema_accuracy[fam] = (
                    self.ema_decay * prev + (1.0 - self.ema_decay) * ep_acc
                )
            self.episode_number += 1
        return next_state, reward, terminated, truncated, info
