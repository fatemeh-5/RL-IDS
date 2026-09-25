"""agent — classic RL library for NIDS Zero-Day detection.

Layout
------
envs/         Gymnasium environments
models/       Q-networks
buffers/      Replay memory (uniform + PER)
training/     B0–B10 loops + advanced/ (PPO, Rainbow, C51, …)
evaluation/   Metrics on Known-Test + Zero-Day
data/         Constants + leakage-safe dataset pipeline
utils/        YAML config + checkpointing
runners/      Catalog and train orchestration (code)

Results CSVs live in Codes/experiments/, not here.
"""

__version__ = "0.7.0-advance"
__all__ = ["__version__"]