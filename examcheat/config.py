from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs" / "default.yaml"


def _set_dotted(cfg: dict, key: str, value):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    with open(path or DEFAULT_CFG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k, yaml.safe_load(v))
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else Path.cwd() / p


def clone(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
