"""``is_done`` must not mistake a short smoke for a finished sweep run.

Smoke runs write the SAME run_id and their own TERMINAL.json, so a bare
existence check would let the sweep skip those cells and silently leave the audit
with 20-epoch data where it believes it has 120. This locks both conditions:
``smoke`` false AND the configured epoch count.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_sweep import enumerate_runs, is_done   # noqa: E402

RUN_ID = "cifar10_symmetric0.4_ce_seed0"
FULL_EPOCHS = 120


def _write_terminal(root, run_id: str, **fields) -> str:
    d = os.path.join(root, run_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "TERMINAL.json"), "w") as fh:
        json.dump(dict(status="completed", **fields), fh)
    return str(root)


def test_absent_terminal_is_not_done(tmp_path):
    os.makedirs(os.path.join(tmp_path, RUN_ID))
    assert is_done(str(tmp_path), RUN_ID, FULL_EPOCHS) is False


def test_smoke_terminal_is_not_done(tmp_path):
    root = _write_terminal(tmp_path, RUN_ID, epochs=20, smoke=True)
    assert is_done(root, RUN_ID, FULL_EPOCHS) is False


def test_short_run_is_not_done_even_when_not_flagged_smoke(tmp_path):
    root = _write_terminal(tmp_path, RUN_ID, epochs=20, smoke=False)
    assert is_done(root, RUN_ID, FULL_EPOCHS) is False


def test_full_length_non_smoke_run_is_done(tmp_path):
    root = _write_terminal(tmp_path, RUN_ID, epochs=FULL_EPOCHS, smoke=False)
    assert is_done(root, RUN_ID, FULL_EPOCHS) is True


def test_full_length_run_still_flagged_smoke_is_not_done(tmp_path):
    root = _write_terminal(tmp_path, RUN_ID, epochs=FULL_EPOCHS, smoke=True)
    assert is_done(root, RUN_ID, FULL_EPOCHS) is False


@pytest.mark.parametrize("body", ["", "{", '{"status": "completed", "epochs":'])
def test_unreadable_terminal_is_not_done(tmp_path, body):
    d = os.path.join(tmp_path, RUN_ID)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "TERMINAL.json"), "w") as fh:
        fh.write(body)
    assert is_done(str(tmp_path), RUN_ID, FULL_EPOCHS) is False


def test_epochs_none_keeps_the_old_existence_only_behaviour(tmp_path):
    root = _write_terminal(tmp_path, RUN_ID, epochs=20, smoke=True)
    assert is_done(root, RUN_ID) is True


def test_sweep_enumerates_the_pre_registered_48_runs():
    cfg = dict(datasets=["cifar10", "cifar100"],
               noise_configs=[{"type": "symmetric", "eta": 0.2}, {"type": "symmetric", "eta": 0.4},
                              {"type": "symmetric", "eta": 0.6}, {"type": "asymmetric", "eta": 0.4}],
               learners={"ce": {}, "elr": {}}, seeds=[0, 1, 2])
    runs = enumerate_runs(cfg)
    assert len(runs) == 48
    assert len({r.run_id for r in runs}) == 48
    # 16 pre-registered (dataset x noise x learner) cells, 3 seeds each
    cells = {(r.dataset, r.noise_type, r.eta, r.learner) for r in runs}
    assert len(cells) == 16
