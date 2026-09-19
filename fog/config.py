#!/usr/bin/env python3
"""
fog/config.py — read the experiment YAML/JSON without exploding.

Fog is handed the same config file the launcher used. A missing or malformed
config must never take down the ground station mid-flight, so every failure
here degrades to {} and prints a warning: the run still records coverage, it
just loses the fire model and the tag.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            text = f.read()
    except Exception as e:
        print(f"[fog] WARN: could not read config '{path}': {e}")
        return {}

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".json":
            data = json.loads(text)
        elif ext in (".yaml", ".yml"):
            if yaml is None:
                print("[fog] WARN: YAML config given but PyYAML is not installed "
                      "(pip install pyyaml). Continuing without config fields.")
                return {}
            data = yaml.safe_load(text)
        else:
            print(f"[fog] WARN: unknown config extension '{ext}'; expected .yaml/.yml/.json")
            return {}
    except Exception as e:
        print(f"[fog] WARN: failed to parse config '{path}': {e}")
        return {}

    return data if isinstance(data, dict) else {}


def resolve_tag(cli_tag: str, cfg: Dict[str, Any], cfg_path: str) -> str:
    """Preference order: CLI flag, then config's experiment_tag, then the
    config filename, then 'exp'."""
    if cli_tag and cli_tag.strip():
        return cli_tag.strip()
    cfg_tag = str(cfg.get("experiment_tag", "") or "")
    if cfg_tag:
        return cfg_tag
    if cfg_path:
        return os.path.splitext(os.path.basename(cfg_path))[0]
    return "exp"