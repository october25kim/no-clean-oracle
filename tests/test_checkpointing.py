"""Scheduled saves must leave one DISTINCT file per saved epoch.

G1's original runs wrote every scheduled save to a single ``checkpoint.pt``, so 12
saves left 1 file and the selection trajectory was lost. A checkpoint-selection audit
cannot recover from that after the fact, so it is locked here.

torch lives only in the training container; this module skips on the analysis host.
The image has no pytest -- to run it there:
  docker run --rm --user 1000:1000 -v <repo>:<repo> -w <repo> fedcore-c400r:latest \
      sh -c "pip install -q pytest && python -m pytest tests/test_checkpointing.py -q"
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn   # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from train.trainer import CELoss, Trainer, TrainConfig   # noqa: E402


def _trainer(run_dir, epochs=20, save_every=5):
    cfg = TrainConfig(epochs=epochs, lr=0.01, momentum=0.9, weight_decay=0.0,
                      batch_size=4, lr_schedule="cosine", save_every=save_every)
    model = nn.Linear(3, 2)
    return Trainer(model, CELoss(), cfg, "cpu", str(run_dir), {"run_id": "t"},
                   eval_fn=lambda m: {})


def _scheduled_epochs(epochs: int, save_every: int):
    return [e for e in range(epochs) if (e + 1) % save_every == 0 or e == epochs - 1]


def test_n_scheduled_saves_yield_n_distinct_files(tmp_path):
    epochs, save_every = 20, 5
    t = _trainer(tmp_path, epochs, save_every)
    scheduled = _scheduled_epochs(epochs, save_every)
    assert len(scheduled) == 4

    for e in scheduled:
        t._save_ckpt(e)

    files = sorted(f for f in os.listdir(tmp_path) if f.startswith("checkpoint_ep"))
    assert files == [f"checkpoint_ep{e:03d}.pt" for e in scheduled]
    assert len(files) == len(scheduled)                    # N saves -> N files
    # distinct paths AND distinct inodes from each other
    inodes = {os.stat(os.path.join(tmp_path, f)).st_ino for f in files}
    assert len(inodes) == len(files)


def test_every_saved_epoch_is_recoverable_and_holds_its_own_state(tmp_path):
    t = _trainer(tmp_path, epochs=20, save_every=5)
    seen = {}
    for e in _scheduled_epochs(20, 5):
        with torch.no_grad():                              # make each epoch's state unique
            t.model.weight.fill_(float(e))
        t._save_ckpt(e)
        seen[e] = float(e)
    for e, w in seen.items():
        ck = torch.load(t.epoch_ckpt(e), map_location="cpu")
        assert ck["epoch"] == e
        assert torch.allclose(ck["model"]["weight"], torch.full((2, 3), w))


def test_latest_tracks_the_newest_epoch_without_consuming_the_history(tmp_path):
    t = _trainer(tmp_path, epochs=20, save_every=5)
    for e in _scheduled_epochs(20, 5):
        t._save_ckpt(e)

    assert os.path.isfile(t.latest)
    assert torch.load(t.latest, map_location="cpu")["epoch"] == 19
    # latest.pt is an alias of the newest epoch file, not a 13th independent copy
    assert os.stat(t.latest).st_ino == os.stat(t.epoch_ckpt(19)).st_ino
    assert not os.path.exists(t.legacy_ckpt)               # the old shared path is gone
    assert len([f for f in os.listdir(tmp_path) if f.startswith("checkpoint_ep")]) == 4


def test_resume_prefers_latest_and_still_accepts_a_legacy_checkpoint(tmp_path):
    t = _trainer(tmp_path, epochs=20, save_every=5)
    t._save_ckpt(9)
    fresh = _trainer(tmp_path, epochs=20, save_every=5)
    fresh.maybe_resume()
    assert fresh.start_epoch == 10

    # a G1-era run dir has only the shared-path file
    legacy_dir = tmp_path / "legacy"
    os.makedirs(legacy_dir)
    old = _trainer(legacy_dir, epochs=20, save_every=5)
    torch.save(dict(model=old.model.state_dict(), opt=old.opt.state_dict(), epoch=7,
                    prev_pred=None), old.legacy_ckpt)
    resumed = _trainer(legacy_dir, epochs=20, save_every=5)
    resumed.maybe_resume()
    assert resumed.start_epoch == 8


def test_no_temp_files_survive_a_save(tmp_path):
    t = _trainer(tmp_path, epochs=20, save_every=5)
    for e in _scheduled_epochs(20, 5):
        t._save_ckpt(e)
    assert [f for f in os.listdir(tmp_path) if ".tmp." in f] == []
