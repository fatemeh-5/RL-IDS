"""Prioritized experience replay (same logic as Untitled B4/B6)."""

from __future__ import annotations

import numpy as np


class PrioritizedReplayBuffer:
    """Experiences with larger TD errors are sampled more often."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-6,
        random_state: int = 42,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        if priority_epsilon <= 0:
            raise ValueError("priority_epsilon must be greater than zero.")

        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.rng = np.random.default_rng(random_state)

    def add(self, state, action, reward, next_state, done) -> None:
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self.buffer) == 0:
            max_priority = 1.0
        else:
            max_priority = float(self.priorities[: len(self.buffer)].max())
            if max_priority <= 0:
                max_priority = 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if batch_size > len(self.buffer):
            raise ValueError("batch_size cannot exceed current buffer size.")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be between 0 and 1.")

        current_priorities = self.priorities[: len(self.buffer)]
        scaled_priorities = current_priorities**self.alpha
        probability_sum = scaled_priorities.sum()
        if probability_sum <= 0:
            probabilities = np.full(
                len(self.buffer), 1.0 / len(self.buffer), dtype=np.float32
            )
        else:
            probabilities = scaled_priorities / probability_sum

        indices = self.rng.choice(
            len(self.buffer), size=batch_size, replace=False, p=probabilities
        )
        minibatch = [self.buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = zip(*minibatch)

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            indices.astype(np.int64),
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        td_errors = np.asarray(td_errors, dtype=np.float32)
        if len(indices) != len(td_errors):
            raise ValueError("indices and td_errors must have equal lengths.")
        self.priorities[indices] = np.abs(td_errors) + self.priority_epsilon

    def __len__(self) -> int:
        return len(self.buffer)


def calculate_per_beta(
    episode: int,
    total_episodes: int,
    beta_start: float = 0.40,
    beta_end: float = 1.00,
) -> float:
    if total_episodes <= 1:
        return float(beta_end)
    progress = episode / (total_episodes - 1)
    beta = beta_start + progress * (beta_end - beta_start)
    return float(np.clip(beta, beta_start, beta_end))
