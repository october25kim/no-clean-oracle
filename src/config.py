"""Config loading with an ``extends:`` key.

A campaign variant should state only what it changes. Copying base.yaml and editing
two lines invites silent drift between the copy and the original, and for the B2 rerun
that drift would be indistinguishable from a real trajectory difference -- so the
variant inherits instead.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge; non-dict values in ``override`` win outright."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, _seen: tuple = ()) -> Dict[str, Any]:
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"circular config extends chain at {path}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(os.path.join(os.path.dirname(path), parent), _seen + (path,))
        cfg = deep_merge(base, cfg)
    return cfg
