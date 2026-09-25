#!/usr/bin/env python3
"""Train NIDS-DRL experiments (B0–B10 + advanced algorithms)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.runners.catalog import (
    get_experiment,
    list_advanced_experiments,
    list_baseline_experiments,
    list_experiments,
)
from agent.runners.runners import rebuild_experiment, rebuild_many


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NIDS-DRL experiments (B0–B10 + advanced)."
    )
    parser.add_argument("experiments", nargs="*", help="e.g. B0 B6 PPO RAINBOW")
    parser.add_argument("--all", action="store_true", help="Train all baseline B0–B10")
    parser.add_argument(
        "--all-advanced",
        action="store_true",
        help="Train all registered advanced algorithms",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--mode", choices=["train", "resume"], default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    if args.list or (not args.all and not args.all_advanced and not args.experiments):
        print("Baseline experiments (B0–B10):")
        for name in list_baseline_experiments():
            print(f"  - {name}")
        print("\nAdvanced algorithms (advance-AI):")
        for name in list_advanced_experiments():
            spec = get_experiment(name)
            print(f"  - {name:10s}  {spec.title}")
        print("\nExamples:")
        print("  python scripts/train.py B0")
        print("  python scripts/train.py PPO")
        print("  python scripts/train.py --all")
        print("  python scripts/train.py --all-advanced")
        print("  python scripts/train.py B4 --mode resume")
        return

    if args.all:
        print(json.dumps(
            rebuild_many(
                list_baseline_experiments(),
                mode=args.mode,
                episodes_override=args.episodes,
                continue_on_error=not args.stop_on_error,
            ),
            indent=2,
        ))
        return

    if args.all_advanced:
        print(json.dumps(
            rebuild_many(
                list_advanced_experiments(),
                mode=args.mode,
                episodes_override=args.episodes,
                continue_on_error=not args.stop_on_error,
            ),
            indent=2,
        ))
        return

    if len(args.experiments) == 1:
        rebuild_experiment(args.experiments[0], mode=args.mode,
                           episodes_override=args.episodes)
        return

    print(json.dumps(
        rebuild_many(
            [e.upper().replace("-", "_") for e in args.experiments],
            mode=args.mode,
            episodes_override=args.episodes,
            continue_on_error=not args.stop_on_error,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()