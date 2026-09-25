#!/usr/bin/env python3
"""One-shot snapshot of the multi-seed sweep: what is running right now.

Shows: orchestrator alive?, cells done/failed/total, the CURRENT cell (model,
feature set, seed, episode x/N, per-episode time, cell ETA), the last few
finished cells, and an overall sweep ETA. Read-only; safe to run any time.

Usage:
    python scripts/sweep_status.py            # snapshot
    python scripts/sweep_status.py --tail 20  # + last 20 lines of progress.log

Live view of the progress log (PowerShell):
    Get-Content experiments\\MULTISEED\\progress.log -Wait -Tail 20
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_multiseed_sweep import DEFAULT_SEEDS, build_plan  # noqa: E402

MULTISEED = ROOT / "experiments" / "MULTISEED"
PID_FILE = MULTISEED / "sweep.pid"
SWEEP_LOG = MULTISEED / "sweep_log.jsonl"
PROGRESS_LOG = MULTISEED / "progress.log"


def fmt(seconds: float) -> str:
    seconds = max(0, int(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return f"{d}d{h:02d}h{m:02d}m"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m"


def orchestrator_alive() -> str:
    if not PID_FILE.exists():
        return "unknown (no sweep.pid)"
    pid = PID_FILE.read_text().strip().splitlines()[0] if PID_FILE.read_text().strip() else ""
    if not pid:
        return "unknown (empty sweep.pid)"
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            alive = "python" in out.lower()
        else:
            os.kill(int(pid), 0)
            alive = True
    except Exception:  # noqa: BLE001
        alive = False
    return f"RUNNING (pid {pid})" if alive else f"NOT RUNNING (stale pid {pid})"


def read_history(ckpt_dir: Path) -> tuple[int | None, float | None]:
    """(episodes checkpointed, mean seconds/episode) from a cell's checkpoint."""
    hist = ckpt_dir / "training_history.csv"
    if not hist.exists():
        return None, None
    with hist.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0, None
    key = next((k for k in rows[0] if k.lower() in ("episode_time_seconds", "seconds")), None)
    secs = [float(r[key]) for r in rows if key and r.get(key)]
    return len(rows), (sum(secs) / len(secs) if secs else None)


def last_progress_line(label_tokens: tuple[str, ...]) -> str | None:
    if not PROGRESS_LOG.exists():
        return None
    lines = PROGRESS_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if all(tok in line for tok in label_tokens):
            return line
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot of the multi-seed sweep.")
    parser.add_argument("--tail", type=int, default=0, help="Also print last N progress.log lines.")
    args = parser.parse_args()

    plan = build_plan(DEFAULT_SEEDS)
    done = [c for c in plan if (c["out_dir"] / "METRICS_ROW.json").exists()]
    failed = [c for c in plan if (c["out_dir"] / "FAILED.json").exists()
              and not (c["out_dir"] / "METRICS_ROW.json").exists()]
    todo = [c for c in plan if not (c["out_dir"] / "METRICS_ROW.json").exists()]

    records = []
    if SWEEP_LOG.exists():
        records = [json.loads(l) for l in SWEEP_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"=== Multi-seed sweep status  ({datetime.now():%Y-%m-%d %H:%M:%S}) ===")
    print(f"Orchestrator : {orchestrator_alive()}")
    print(f"Cells        : {len(done)}/{len(plan)} done, {len(failed)} failed, {len(todo)} remaining")

    # Current cell = first not-done cell in plan order whose out_dir exists
    # (orchestrator creates out_dir right before launching the cell).
    current = next((c for c in todo if (c["out_dir"] / "log.txt").exists()
                    and not (c["out_dir"] / "FAILED.json").exists()), None)
    print()
    if current is None:
        print("Current cell : none in progress")
    else:
        log_txt = current["out_dir"] / "log.txt"
        started = log_txt.stat().st_ctime
        print(f"Current cell : {current['kind']} {current['config']} | features={current['feature_set']} "
              f"| seed={current['seed']}")
        print(f"  started    : {datetime.fromtimestamp(started):%Y-%m-%d %H:%M} "
              f"({fmt(time.time() - started)} ago)")
        line = last_progress_line((f"{current['config']} ", current["feature_set"], f"seed{current['seed']}"))
        if line:
            print(f"  latest     : {line}")
        elif current["kind"] == "drl":
            # cell started before progress logging existed: fall back to checkpoint
            n_eps, per_ep = read_history(current["out_dir"] / "checkpoints")
            total_eps = 31
            if n_eps is not None:
                ckpt_t = (current["out_dir"] / "checkpoints" / "training_history.csv").stat().st_mtime
                msg = f"  checkpoint : {n_eps}/{total_eps} episodes saved (at {datetime.fromtimestamp(ckpt_t):%H:%M})"
                if per_ep:
                    est_now = min(total_eps, n_eps + int((time.time() - ckpt_t) / per_ep))
                    msg += (f" | ~{per_ep:.0f}s/episode | est. now on ep ~{est_now}/{total_eps}"
                            f" | cell ETA ~{fmt(per_ep * (total_eps - est_now))} (+eval)")
                print(msg)
            else:
                print("  checkpoint : none yet (still in first episodes / data loading)")

    if records:
        print("\nLast finished cells:")
        for r in records[-5:]:
            print(f"  {datetime.fromtimestamp(r['ts']):%m-%d %H:%M}  {r['status']:6s}  "
                  f"{r['label']:32s} {fmt(r['elapsed_s'])}")
        ok_times = [r["elapsed_s"] for r in records if r["status"] == "ok"]
        drl_times = [r["elapsed_s"] for r in records if r["status"] == "ok" and r["label"].startswith("drl/")]
        sup_times = [r["elapsed_s"] for r in records if r["status"] == "ok" and r["label"].startswith("supervised/")]
        if ok_times:
            avg_drl = sum(drl_times) / len(drl_times) if drl_times else sum(ok_times) / len(ok_times)
            avg_sup = sum(sup_times) / len(sup_times) if sup_times else avg_drl
            n_drl = sum(1 for c in todo if c["kind"] == "drl")
            n_sup = len(todo) - n_drl
            eta = avg_drl * n_drl + avg_sup * n_sup
            note = "" if sup_times else " (supervised cells assumed as slow as DRL until one finishes)"
            print(f"\nSweep ETA    : ~{fmt(eta)} -> {datetime.fromtimestamp(time.time() + eta):%Y-%m-%d %H:%M}{note}")

    if args.tail and PROGRESS_LOG.exists():
        print(f"\n--- last {args.tail} lines of {PROGRESS_LOG.relative_to(ROOT)} ---")
        for line in PROGRESS_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-args.tail:]:
            print(line)


if __name__ == "__main__":
    main()
