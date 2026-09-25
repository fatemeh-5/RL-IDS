"""Human-readable sweep progress log.

Every sweep cell (one subprocess per config/feature_set/seed) tees its stdout
through `ProgressTee`. Everything still goes to the cell's own log.txt, and the
lines that matter (per-episode training lines, supervised results) are also
appended to ONE shared file, experiments/MULTISEED/progress.log, prefixed with
timestamp + which model/feature_set/seed produced them. The TensorFlow noise
in log.txt never reaches progress.log.

    2026-09-25 23:31:02 | B1    ports  seed42 | ep 15/31 | Reward: 146.0 | Acc: 0.785 | ... | cell ETA 34m

Logging only: nothing here touches any RNG, so determinism is unaffected.
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROGRESS_LOG = ROOT / "experiments" / "MULTISEED" / "progress.log"

# Trainer lines look like "Episode 07/30 | Reward: ... " (0-based episode index).
_EPISODE_RE = re.compile(r"^Episode\s+(\d+)\s*/\s*(\d+)\s*\|\s*(.*)$")
# Supervised baseline result / skip / error lines, e.g. "[rf] Known-Test Recall=...".
_SUPERVISED_RE = re.compile(r"^\[(?:skip |error )?[a-z]+\]")


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class ProgressTee:
    """Wraps sys.stdout: passes everything through (line-buffered) and mirrors
    the interesting lines into the shared progress log."""

    def __init__(self, stream, label: str, log_path: Path = PROGRESS_LOG):
        self._stream = stream
        self._label = label
        self._log_path = log_path
        self._buf = ""
        self._t0 = time.time()
        self._train_t0: float | None = None
        self._episode_secs: list[float] = []

    # -- shared log ---------------------------------------------------------
    def event(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} | {self._label} | {message}\n"
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass  # progress logging must never kill a training cell

    def elapsed(self) -> str:
        return _fmt_duration(time.time() - self._t0)

    def _handle_line(self, line: str) -> None:
        text = line.strip()
        match = _EPISODE_RE.match(text)
        if match:
            ep, last, rest = int(match.group(1)), int(match.group(2)), match.group(3)
            t_match = re.search(r"t[:=]\s*([\d.]+)s", rest)
            if t_match:
                self._episode_secs.append(float(t_match.group(1)))
            if self._train_t0 is None:
                self._train_t0 = time.time() - (self._episode_secs[-1] if self._episode_secs else 0.0)
            done = ep + 1
            total = last + 1
            if self._episode_secs:
                per_ep = sum(self._episode_secs) / len(self._episode_secs)
            else:
                per_ep = (time.time() - self._train_t0) / done
            eta = per_ep * (total - done)
            self.event(f"ep {done:>2d}/{total} | {rest} | cell ETA {_fmt_duration(eta)} (+eval)")
        elif _SUPERVISED_RE.match(text) or text.startswith("Resuming"):
            self.event(text)

    # -- file-like interface -----------------------------------------------
    def write(self, data: str) -> int:
        n = self._stream.write(data)
        self._stream.flush()
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle_line(line)
        return n

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install(label: str) -> ProgressTee:
    """Replace sys.stdout with a ProgressTee and return it (for .event())."""
    tee = ProgressTee(sys.stdout, label)
    sys.stdout = tee
    return tee


def cell_label(config: str, feature_set: str, seed: int) -> str:
    return f"{config:<5s} {feature_set:<7s} seed{seed}"
