#!/usr/bin/env python3
"""Build or reuse cache/prepared after a reboot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.data.pipeline import cache_is_ready, prepare_and_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare scaled (+ optional SMOTE) data cache for RL-IDS."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cache even if it already exists.",
    )
    parser.add_argument(
        "--skip-smote",
        action="store_true",
        help="Only cache scaled splits (faster; enough for eval / smoke tests).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional subsample for quick smoke tests.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset CSV path.",
    )
    args = parser.parse_args()

    if not args.force and cache_is_ready(require_balanced=not args.skip_smote):
        print("Cache already ready. Use --force to rebuild.")
        return

    prepare_and_cache(
        dataset_path=args.dataset,
        max_rows=args.max_rows,
        skip_smote=args.skip_smote,
        force=args.force,
    )


if __name__ == "__main__":
    main()
