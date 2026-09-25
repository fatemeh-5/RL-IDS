"""B0–B10 experiment runners (code) — not result CSVs.

Result CSVs live in Codes/experiments/.
This package only knows *how* to train/evaluate each Bi.
"""

from agent.runners.catalog import CATALOG, get_experiment, list_experiments

__all__ = ["CATALOG", "get_experiment", "list_experiments"]
