"""YAML experiment config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent.data.constants import PROJECT_ROOT


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    cfg["_config_path"] = str(config_path.resolve())
    return cfg


def resolve_run_dir(cfg: dict[str, Any]) -> Path:
    run_dir = Path(cfg.get("run_dir", f"experiments/{cfg['experiment_id']}"))
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
