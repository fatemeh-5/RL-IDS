"""Experience replay buffers."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Uniform replay buffer."""

    def __init__(self, capacity: int, random_state: int = 42):
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero.")
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.random = random.Random(random_state)

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            raise ValueError("batch_size cannot exceed current buffer size.")
        minibatch = self.random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# Re-export PER for a single import surface.
from agent.buffers.prioritized import PrioritizedReplayBuffer, calculate_per_beta  # noqa: E402
