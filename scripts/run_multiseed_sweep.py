#!/usr/bin/env python3
"""Orchestrate the full B0-B10 + RF/XGB/MLP/LSTM x ports/noports x 8-seed sweep.

Design goals:
  - Resumable: a cell counts as done iff its METRICS_ROW.json exists. Safe to
    kill (Ctrl-C, crash, machine reboot) and re-run this script verbatim; it
    picks up wherever it left off, no double work.
  - Isolated: each cell is a SEPARATE subprocess (run_drl_cell.py or
    supervised_baseline.py). A crash/OOM/NaN in one cell can't corrupt or
    kill the other ~230 cells, and TF/Keras global state never accumulates
    across cells in one process over a multi-day run.
  - Breadth-first over seeds: loop order is seed -> feature_set -> (configs,
    then supervised models), so an interrupted sweep leaves the *most*
    complete coverage across configs at the fewest seeds, rather than 8
    seeds of B0 and nothing else.
  - Legible progress: a compact one-line-per-cell JSONL ledger at
    experiments/MULTISEED/sweep_log.jsonl, plus running ETA on stdout. Each
    cell's full training stdout/stderr goes to its own out_dir/log.txt so it
    doesn't flood the orchestrator log.

Usage:
    python scripts/run_multiseed_sweep.py
    python scripts/run_multiseed_sweep.py --seeds 42 43 44   # override seed list
    python scripts/run_multiseed_sweep.py --dry-run          # just print the plan
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON = sys.executable

DRL_CONFIGS = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
SUP_MODELS = ["rf", "xgb", "mlp", "lstm"]
FEATURE_SETS = ["ports", "noports"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]

RUNS_ROOT = ROOT / "experiments" / "MULTISEED" / "runs"
SWEEP_LOG = ROOT / "experiments" / "MULTISEED" / "sweep_log.jsonl"

# Ground truth for the trainer-fidelity preflight check: B0 is the softmax-CE
# baseline trainer, B1-B3 uniform-replay Double-DQN, B4-B6 PER, B7-B10 PER
# with the attack-family env, B11 PER with the investigate env (3 actions).
# Every config must resolve to ITS OWN trainer from configs/<ID>.yaml, never a
# trainer shared/leaked from another config.
EXPECTED_TRAINER = {
    "B0": "baseline",
    "B1": "double", "B2": "double", "B3": "double",
    "B4": "per", "B5": "per", "B6": "per",
    "B7": "per", "B8": "per", "B9": "per", "B10": "per", "B11": "per",
}
EXPECTED_FAMILY_ENV = {"B7", "B8", "B9", "B10"}
EXPECTED_INVESTIGATE_ENV = {"B11"}


def print_trainer_fidelity_table(configs: list[str]) -> None:
    """Print config -> (trainer, model builder, env) resolved fresh from each
    config's YAML, and hard-fail the sweep if any config doesn't resolve to
    its expected trainer (see EXPECTED_TRAINER) — the check requested to guard
    against B0 silently training with the Double/PER trainer."""
    from agent.runners.runners import describe_trainer

    rows = [describe_trainer(cfg_id) for cfg_id in configs]

    print("\nTrainer fidelity check (resolved fresh from each configs/<ID>.yaml):")
    header = f"  {'Config':7s} {'Trainer':10s} {'Model builder':20s} {'Env':14s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(f"  {r['config']:7s} {r['trainer']:10s} {r['model_builder']:20s} {r['env']:14s}")

    problems = []
    for r in rows:
        cfg_id = r["config"]
        expected = EXPECTED_TRAINER.get(cfg_id)
        if expected is not None and r["trainer"] != expected:
            problems.append(
                f"{cfg_id}: expected trainer={expected!r}, resolved trainer={r['trainer']!r}"
            )
        if cfg_id in EXPECTED_FAMILY_ENV and r["env"] != "family":
            problems.append(f"{cfg_id}: expected env='family', resolved env={r['env']!r}")
        if cfg_id in EXPECTED_INVESTIGATE_ENV and r["env"] != "investigate":
            problems.append(f"{cfg_id}: expected env='investigate', resolved env={r['env']!r}")

    if problems:
        print("\nTRAINER FIDELITY BUG DETECTED — refusing to run the sweep:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("  OK: every config resolves to its own trainer/model/env. Safe to train.\n")


def build_plan(seeds: list[int]) -> list[dict]:
    """Breadth-first: seed -> feature_set -> (drl configs, then sup models)."""
    plan = []
    for seed in seeds:
        for fs in FEATURE_SETS:
            for cfg in DRL_CONFIGS:
                out_dir = RUNS_ROOT / "drl" / cfg / fs / f"seed{seed}"
                plan.append(
                    {
                        "kind": "drl",
                        "config": cfg,
                        "feature_set": fs,
                        "seed": seed,
                        "out_dir": out_dir,
                    }
                )
            for model in SUP_MODELS:
                out_dir = RUNS_ROOT / "supervised" / model / fs / f"seed{seed}"
                plan.append(
                    {
                        "kind": "supervised",
                        "config": model,
                        "feature_set": fs,
                        "seed": seed,
                        "out_dir": out_dir,
                    }
                )
    return plan


def cell_command(cell: dict) -> list[str]:
    if cell["kind"] == "drl":
        return [
            PYTHON, str(ROOT / "scripts" / "run_drl_cell.py"),
            "--config", cell["config"],
            "--feature-set", cell["feature_set"],
            "--seed", str(cell["seed"]),
            "--out-dir", str(cell["out_dir"]),
        ]
    return [
        PYTHON, str(ROOT / "scripts" / "supervised_baseline.py"),
        "--models", cell["config"],
        "--feature-set", cell["feature_set"],
        "--seed", str(cell["seed"]),
        "--out-dir", str(cell["out_dir"]),
    ]


def append_log(record: dict) -> None:
    SWEEP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SWEEP_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _deterministic_env() -> dict:
    # Set once here (parent process, before each child interpreter starts) rather
    # than inside the child script: PYTHONHASHSEED only takes effect if present in
    # the environment *before* the interpreter boots, so setting it from within the
    # child after it's already running would be a no-op for that same process.
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["TF_DETERMINISTIC_OPS"] = "1"
    env["TF_CUDNN_DETERMINISTIC"] = "1"
    return env


def run_cell(cell: dict) -> tuple[bool, float]:
    out_dir = cell["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"
    cmd = cell_command(cell)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(cmd) + "\n\n")
        log_handle.flush()
        proc = subprocess.run(
            cmd, cwd=str(ROOT), stdout=log_handle, stderr=subprocess.STDOUT,
            env=_deterministic_env(),
        )
    elapsed = time.time() - t0
    ok = proc.returncode == 0 and (out_dir / "METRICS_ROW.json").exists()
    return ok, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full multi-seed sweep.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print_trainer_fidelity_table(DRL_CONFIGS)

    plan = build_plan(args.seeds)
    total = len(plan)

    todo = [c for c in plan if not (c["out_dir"] / "METRICS_ROW.json").exists()]
    print(f"Sweep plan: {total} cells total, {len(todo)} remaining "
          f"({total - len(todo)} already done).")

    if args.dry_run:
        for c in todo[:20]:
            print(f"  would run: {c['kind']:10s} {c['config']:6s} "
                  f"{c['feature_set']:8s} seed={c['seed']}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return

    done_count = total - len(todo)
    fail_count = 0
    sweep_t0 = time.time()
    cell_durations: list[float] = []

    for i, cell in enumerate(todo, start=1):
        label = f"{cell['kind']}/{cell['config']}/{cell['feature_set']}/seed{cell['seed']}"
        print(f"[{done_count + fail_count + 1}/{total}] START {label}", flush=True)
        t_cell0 = time.time()
        try:
            ok, elapsed = run_cell(cell)
        except KeyboardInterrupt:
            print("Interrupted by user; sweep state is resumable, exiting.")
            raise
        except Exception as exc:  # noqa: BLE001 - keep the sweep alive
            ok, elapsed = False, time.time() - t_cell0
            append_log({
                "ts": time.time(), "label": label, "status": "error",
                "elapsed_s": round(elapsed, 1), "error": str(exc),
            })
        else:
            append_log({
                "ts": time.time(), "label": label,
                "status": "ok" if ok else "failed",
                "elapsed_s": round(elapsed, 1),
            })

        cell_durations.append(elapsed)
        if ok:
            done_count += 1
        else:
            fail_count += 1
            print(f"  [FAILED] {label} — see {cell['out_dir']}/log.txt", flush=True)

        avg = sum(cell_durations) / len(cell_durations)
        remaining = len(todo) - i
        eta_s = avg * remaining
        elapsed_total = time.time() - sweep_t0
        print(
            f"  done in {elapsed / 60:.1f} min | "
            f"progress {done_count + fail_count}/{total} "
            f"(ok={done_count}, failed={fail_count}) | "
            f"elapsed {elapsed_total / 3600:.2f}h | "
            f"ETA {eta_s / 3600:.2f}h",
            flush=True,
        )

    print("\nSweep pass complete.")
    print(f"  ok={done_count}  failed={fail_count}  total={total}")
    if fail_count:
        print("  Re-run this script to retry failed/incomplete cells.")


if __name__ == "__main__":
    main()
