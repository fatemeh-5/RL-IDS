"""B11 environment: investigate action + post-alert queue effect.

Extends FamilyAwareIDSEnvironment with:
  - 3 actions: {0: benign, 1: attack, 2: investigate}
  - State augmented with 2 meta-features (12-D instead of 10-D):
        * investigate budget remaining (0..1, /20)
        * post-alert countdown          (0..1, /5)
  - Investigate budget of 20 per episode
        - Free while budget > 0
        - Costs -2 when budget is exhausted (forces a commit)
        - When investigating, the same flow is shown again (state re-fed)
          with updated meta-features
  - Post-alert queue effect:
        - Every time the agent picks "attack", a 5-flow countdown starts
        - During the countdown, the next flow's benign probability is
          boosted by 30 percentage points (up to 100%). This is a
          function of the ACTION only, NOT the true label — no leakage.
  - Cost-sensitive reward matrix (TP=+5, TN=+2, FP=-3, FN=-10)

Drop this file next to ids_env.py in agent/envs/ and add
    from .investigate_env import InvestigateIDSEnvironment
to agent/envs/__init__.py.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from agent.envs.ids_env import FamilyAwareIDSEnvironment


# Action space
ACTION_BENIGN = 0
ACTION_ATTACK = 1
ACTION_INVESTIGATE = 2

# State augmentation: 2 extra features appended after the 10 flow features
N_META_FEATURES = 2


class InvestigateIDSEnvironment(FamilyAwareIDSEnvironment):
    """Family-aware IDS env with an investigate action and post-alert queue effect."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        attack_names: np.ndarray,
        *,
        # Inherited family-aware settings — same defaults as B10
        mode: str = "restrained",
        episode_length: int = 256,
        benign_per_episode: int = 128,
        natural_attack_per_episode: int = 64,
        stratified_attack_per_episode: int = 64,
        adaptive_mix: float = 0.40,
        ema_decay: float = 0.90,
        min_family_quota: int = 2,
        max_family_quota: int = 5,
        random_state: int = 42,
        # NEW: investigate + queue-effect settings
        investigate_budget: int = 20,
        investigate_penalty_exhausted: float = -2.0,
        post_alert_countdown_length: int = 5,
        post_alert_benign_boost: float = 0.30,
        # Cost-sensitive reward matrix (aligns with B5/B6)
        reward_tp: float = 5.0,
        reward_tn: float = 2.0,
        reward_fp: float = -3.0,
        reward_fn: float = -10.0,
    ):
        super().__init__(
            features=features,
            labels=labels,
            attack_names=attack_names,
            mode=mode,
            episode_length=episode_length,
            benign_per_episode=benign_per_episode,
            natural_attack_per_episode=natural_attack_per_episode,
            stratified_attack_per_episode=stratified_attack_per_episode,
            adaptive_mix=adaptive_mix,
            ema_decay=ema_decay,
            min_family_quota=min_family_quota,
            max_family_quota=max_family_quota,
            random_state=random_state,
        )

        # --- override action and observation spaces ---
        self.action_space = spaces.Discrete(3)
        # State grows from n_features to n_features + N_META_FEATURES
        self.n_features_flow = self.n_features
        self.n_features = self.n_features_flow + N_META_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features,), dtype=np.float32,
        )

        # --- investigate + queue-effect config ---
        self.investigate_budget_max = int(investigate_budget)
        self.investigate_penalty_exhausted = float(investigate_penalty_exhausted)
        self.post_alert_countdown_length = int(post_alert_countdown_length)
        self.post_alert_benign_boost = float(post_alert_benign_boost)

        # cost-sensitive reward
        self.reward_tp = float(reward_tp)
        self.reward_tn = float(reward_tn)
        self.reward_fp = float(reward_fp)
        self.reward_fn = float(reward_fn)

        # --- per-episode state ---
        self._budget_remaining = self.investigate_budget_max
        self._post_alert_countdown = 0
        self._last_action_was_investigate = False

        # metrics telemetry (helpful for plots later)
        self._episode_investigations = 0
        self._episode_investigations_useful = 0
        self._episode_attack_alerts = 0

    # ------------------------------------------------------------------ #
    # State augmentation                                                  #
    # ------------------------------------------------------------------ #
    def _augment_state(self, flow_features: np.ndarray) -> np.ndarray:
        """Append (budget_remaining / max, countdown / length) to the flow features."""
        budget_norm = self._budget_remaining / self.investigate_budget_max
        countdown_norm = self._post_alert_countdown / self.post_alert_countdown_length
        meta = np.array([budget_norm, countdown_norm], dtype=np.float32)
        return np.concatenate([flow_features.astype(np.float32), meta])

    # ------------------------------------------------------------------ #
    # Episode construction — reuse parent, but keep a pointer so we can  #
    # substitute the next index when the queue effect fires              #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        self._budget_remaining = self.investigate_budget_max
        self._post_alert_countdown = 0
        self._last_action_was_investigate = False
        self._episode_investigations = 0
        self._episode_investigations_useful = 0
        self._episode_attack_alerts = 0
        info.update({
            "budget_remaining": self._budget_remaining,
            "post_alert_countdown": self._post_alert_countdown,
        })
        return self._augment_state(state), info

    # ------------------------------------------------------------------ #
    # Post-alert queue effect: draw a biased benign/attack replacement    #
    # for a specific upcoming step index                                  #
    # ------------------------------------------------------------------ #
    def _apply_queue_effect(self, upcoming_step: int) -> None:
        """Rewrite episode_indices[upcoming_step] with a benign-boosted draw.

        Only touches the next single upcoming step. Called every time we need
        to look at the following flow while the countdown is active. The
        transition is a function of the AGENT'S ACTION, not the true label —
        no ground-truth information is leaked into the state.
        """
        if upcoming_step >= self.episode_length:
            return
        p_benign = 0.5 + self.post_alert_benign_boost   # baseline 50% + boost
        p_benign = min(1.0, max(0.0, p_benign))
        if self.rng.random() < p_benign:
            pool = self.benign_indices
        else:
            pool = self.attack_indices
        if len(pool):
            self.episode_indices[upcoming_step] = int(
                self.rng.choice(pool, size=1)[0]
            )

    # ------------------------------------------------------------------ #
    # Episode-end bookkeeping — shared by every branch of step() so the   #
    # family-EMA update and telemetry always run once per episode, even  #
    # when the very last step happens to be an investigate action.       #
    # ------------------------------------------------------------------ #
    def _finalize_episode(self, info: dict) -> None:
        for fam, outcomes in self._last_episode_family_correct.items():
            if not outcomes:
                continue
            ep_acc = float(np.mean(outcomes))
            prev = self.family_ema_accuracy[fam]
            self.family_ema_accuracy[fam] = (
                self.ema_decay * prev + (1.0 - self.ema_decay) * ep_acc
            )
        self.episode_number += 1
        info["episode_investigations"] = self._episode_investigations
        info["episode_attack_alerts"] = self._episode_attack_alerts

    # ------------------------------------------------------------------ #
    # step()                                                              #
    # ------------------------------------------------------------------ #
    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        sample_index = int(self.episode_indices[self.current_step])
        true_label = int(self.labels[sample_index])
        family = str(self.attack_names[sample_index])

        # -------------------- Investigate branch --------------------
        if action == ACTION_INVESTIGATE:
            self._episode_investigations += 1
            self._last_action_was_investigate = True

            if self._budget_remaining > 0:
                self._budget_remaining -= 1
                # Budget was available -> free investigation. Same flow shown
                # again with updated meta features (budget decremented).
                reward = 0.0
                self.current_step += 1  # investigate still consumes one step
                terminated = self.current_step >= self.episode_length

                if terminated:
                    next_state = np.zeros(self.n_features_flow, dtype=np.float32)
                else:
                    # Re-feed the SAME flow (investigate = look again).
                    # Overwrite next slot to the current sample.
                    self.episode_indices[self.current_step] = sample_index
                    next_state = self.features[sample_index]

                # If a countdown is running, decrement it (queue-effect
                # doesn't reset just because we investigated).
                if self._post_alert_countdown > 0:
                    self._post_alert_countdown -= 1
                    if not terminated:
                        self._apply_queue_effect(self.current_step)
                        next_state = self.features[
                            int(self.episode_indices[self.current_step])
                        ]

                info = {
                    "sample_index": sample_index,
                    "true_label": true_label,
                    "action": ACTION_INVESTIGATE,
                    "correct": False,          # investigate isn't right/wrong
                    "outcome": "INVESTIGATE_FREE",
                    "budget_remaining": self._budget_remaining,
                    "post_alert_countdown": self._post_alert_countdown,
                }
                if terminated:
                    self._finalize_episode(info)
                return (
                    self._augment_state(next_state),
                    reward, terminated, False, info,
                )

            # Budget exhausted -> penalty; forced to commit (auto-classify
            # as whichever the network's greedy pick would be at test time,
            # but here we just mark it wrong and advance).
            reward = self.investigate_penalty_exhausted
            self.current_step += 1
            terminated = self.current_step >= self.episode_length
            if terminated:
                next_state = np.zeros(self.n_features_flow, dtype=np.float32)
            else:
                next_state = self.features[
                    int(self.episode_indices[self.current_step])
                ]
                if self._post_alert_countdown > 0:
                    self._post_alert_countdown -= 1
                    self._apply_queue_effect(self.current_step)
                    next_state = self.features[
                        int(self.episode_indices[self.current_step])
                    ]

            info = {
                "sample_index": sample_index,
                "true_label": true_label,
                "action": ACTION_INVESTIGATE,
                "correct": False,
                "outcome": "INVESTIGATE_BLOCKED",
                "budget_remaining": self._budget_remaining,
                "post_alert_countdown": self._post_alert_countdown,
            }
            if terminated:
                self._finalize_episode(info)
            return (
                self._augment_state(next_state),
                reward, terminated, False, info,
            )

        # -------------------- Commit branch (benign or attack) --------------------
        self._last_action_was_investigate = False

        # Cost-sensitive reward
        if true_label == 1 and action == ACTION_ATTACK:
            reward, outcome = self.reward_tp, "TP"
        elif true_label == 0 and action == ACTION_BENIGN:
            reward, outcome = self.reward_tn, "TN"
        elif true_label == 0 and action == ACTION_ATTACK:
            reward, outcome = self.reward_fp, "FP"
        else:  # true_label == 1 and action == ACTION_BENIGN
            reward, outcome = self.reward_fn, "FN"

        # Trigger post-alert queue effect if agent alerted (regardless of correctness)
        if action == ACTION_ATTACK:
            self._episode_attack_alerts += 1
            self._post_alert_countdown = self.post_alert_countdown_length

        # Advance to next step
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        if terminated:
            next_state = np.zeros(self.n_features_flow, dtype=np.float32)
        else:
            # Countdown ticks down; while active, override the next step's index
            if self._post_alert_countdown > 0:
                self._apply_queue_effect(self.current_step)
                self._post_alert_countdown -= 1
            next_state = self.features[
                int(self.episode_indices[self.current_step])
            ]

        # Family EMA update (inherited behavior for the "correct" family attack)
        if true_label == 1 and family in self._last_episode_family_correct:
            correct = int(action == ACTION_ATTACK)
            self._last_episode_family_correct[family].append(correct)

        info = {
            "sample_index": sample_index,
            "true_label": true_label,
            "action": int(action),
            "correct": bool(
                (true_label == 1 and action == ACTION_ATTACK)
                or (true_label == 0 and action == ACTION_BENIGN)
            ),
            "outcome": outcome,
            "budget_remaining": self._budget_remaining,
            "post_alert_countdown": self._post_alert_countdown,
        }

        if terminated:
            self._finalize_episode(info)

        return (
            self._augment_state(next_state),
            reward, terminated, False, info,
        )