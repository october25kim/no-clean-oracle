"""Tier 1 forward pass: 36 runs x 24 checkpoints -> logits + embeddings.

Derived from ``scripts/forward_b2.py``, which is the certified precedent -- its 360
checkpoints reproduced the training-time metrics at exactly 0.000e+00 deviation. The
shard-summary/consolidate machinery, the atomic writes and the inference configuration
are carried over unchanged so this pass sits in the same numerical frame. Four things
differ, all of them because the campaign differs:

1. Labels come from the SEALED CIFAR-N splits (``data.real_noise.load_split``), not from
   a synthetic transition matrix. The split's sha256, its measured noise rate and its
   mask definition are re-verified on every load, and the sealed clean labels are checked
   against the dataset targets the loader will actually use.
2. ``reference_frame`` is ``"ext_tier1"``: the extension is its own self-contained frame
   per the extension preregistration, not a re-measurement of G1.
3. SOP runs additionally assert, once per checkpoint, that the stored logits are the raw
   f(x) with no sparse term added (``train.sop.assert_raw_logits``). The count is
   reported; the registered expectation is 9 SOP runs x 24 checkpoints = 216.
4. A non-zero integrity deviation ABORTS. B2 recorded deviations and summarised them at
   the end; here the instruction is exactly 0 (trajectory-identity class: the recomputation
   walks the same path the training-time evaluation walked), so a single non-zero value
   stops the shard rather than spending hours producing outputs no consumer may read.

Inference only. No training, no checkpoint is modified, and ``latest.pt`` is never read --
it is a resume artifact, not part of the selection grid.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from config import load_config                          # noqa: E402
from data import datasets as D                          # noqa: E402
from data.ood_pools import load_ood_pools               # noqa: E402
from data.real_noise import load_split                  # noqa: E402
from eval.ood import auroc, energy_score, msp_score     # noqa: E402
from eval.tail import per_class_error, r_tail_dynamic   # noqa: E402
from models.preact_resnet import build_model            # noqa: E402
from provenance import code_stamp                       # noqa: E402
from train.sop import assert_raw_logits                 # noqa: E402
from run_single import gpu_uuid                         # noqa: E402

REFERENCE_FRAME = "ext_tier1"
EVAL_TRANSFORM_DESC = "ToTensor+Normalize (no randomness)"


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: str, obj: dict) -> None:
    """Write via a per-process temp file, then rename -- concurrent shards must never
    leave a half-written summary, nor race on one final name."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def assert_eval_transform_deterministic(transform) -> List[str]:
    """Refuse to run unless the eval transform is ToTensor + Normalize and nothing else.

    The whole pass is claimed to be reproducible without a seed because the eval
    transform carries no randomness. That claim is worth checking rather than asserting:
    a stray RandomCrop here would silently decouple these logits from the training-time
    evaluation, and the integrity check would then be comparing two different things.
    """
    names = [type(t).__name__ for t in getattr(transform, "transforms", [transform])]
    if names != ["ToTensor", "Normalize"]:
        raise SystemExit(f"eval transform is {names}, expected ['ToTensor', 'Normalize']; "
                         f"a stochastic component would invalidate the pass")
    return names


def shard_tag_for(run_ids: List[str]) -> str:
    """Deterministic tag from the runs a shard owns: same shard -> same filename."""
    return hashlib.sha256("|".join(sorted(run_ids)).encode()).hexdigest()[:10]


def consolidate(out_dir: str) -> dict:
    """Merge every shard summary present into one consolidated file.

    Safe to call from each shard as it finishes: it reads whatever shards have landed and
    writes atomically, so the last caller leaves the complete picture.
    """
    shards = []
    for path in sorted(glob.glob(os.path.join(out_dir, "forward_summary_shard_*.json"))):
        with open(path) as fh:
            shards.append(json.load(fh))
    metas = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*", "ep*", "meta.json"))):
        with open(path) as fh:
            metas.append(json.load(fh))
    runs = sorted({m["run_id"] for m in metas})
    devs = [m["max_abs_deviation_vs_log"] for m in metas] or [0.0]
    secs = [m["seconds"] for m in metas] or [0.0]
    sop_metas = [m for m in metas if m["learner"] == "sop"]
    out = dict(
        consolidated_from=[s["shard_tag"] for s in shards],
        n_shards=len(shards), runs=len(runs), checkpoints=len(metas),
        checkpoints_per_run={r: sum(1 for m in metas if m["run_id"] == r) for r in runs},
        splits=sorted({m["split"] for m in metas}),
        learners=sorted({m["learner"] for m in metas}),
        integrity_class="trajectory-identity (same computation path; tolerance exactly 0)",
        max_abs_deviation_vs_log=max(devs),
        n_checkpoints_with_nonzero_deviation=sum(1 for d in devs if d != 0.0),
        sop_raw_logit_assertions=sum(m.get("sop_raw_logit_assertions", 0) for m in metas),
        sop_checkpoints=len(sop_metas),
        sop_assertion_expectation=216,
        batch_size=sorted({m["batch_size"] for m in metas}),
        dataloader_workers=sorted({m["dataloader_workers"] for m in metas}),
        precision=sorted({m["precision"] for m in metas}),
        reference_frame=sorted({m["reference_frame"] for m in metas}),
        gpu_uuids=sorted({m["gpu_uuid"] for m in metas if m.get("gpu_uuid")}),
        torch=sorted({m["torch"] for m in metas}),
        cuda=sorted({m["cuda"] for m in metas}),
        cudnn=sorted({m["cudnn"] for m in metas}),
        cublas_workspace_config=sorted({m["cublas_workspace_config"] for m in metas}),
        eval_transform=sorted({m["eval_transform"] for m in metas}),
        noise_provenance={m["split"]: m["noise_provenance"] for m in metas},
        compute_seconds_total=round(sum(secs), 1),
        per_checkpoint_seconds=dict(mean=round(sum(secs) / max(len(secs), 1), 2),
                                    min=round(min(secs), 2), max=round(max(secs), 2)),
        shard_wall_clock_seconds={s["shard_tag"]: s["wall_clock_seconds"] for s in shards},
    )
    _atomic_json(os.path.join(out_dir, "forward_summary_consolidated.json"), out)
    return out


def write_manifest(out_dir: str) -> Dict[str, object]:
    """MANIFEST.sha256 over every produced file, sorted by repo-relative path.

    The manifest excludes itself and any shard temp file. Its own sha256 is returned so
    one value attests the whole output tree.
    """
    manifest_path = os.path.join(out_dir, "MANIFEST.sha256")
    entries = []
    for dirpath, _dirs, files in os.walk(out_dir):
        for name in files:
            if name == "MANIFEST.sha256" or ".tmp." in name:
                continue
            full = os.path.join(dirpath, name)
            entries.append((os.path.relpath(full, ROOT), full))
    entries.sort(key=lambda e: e[0])
    lines = [f"{_sha256(full)}  {rel}\n" for rel, full in entries]
    tmp = f"{manifest_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.writelines(lines)
    os.replace(tmp, manifest_path)
    return dict(manifest=manifest_path, files=len(entries),
                manifest_sha256=_sha256(manifest_path),
                bytes=sum(os.path.getsize(f) for _r, f in entries))


@torch.no_grad()
def infer(model, loader, device: str, want_feats: bool, assert_first_batch: bool = False):
    """Logits (and optionally the penultimate embedding) for one evaluation set.

    ``assert_first_batch`` runs the SOP raw-f(x) check on the first batch: it recomputes
    ``model(x)`` and demands bitwise equality with the logits about to be written, which
    is what proves no sparse term u^2 - v^2 was folded in. Once per checkpoint is enough
    -- the property is of the writing code path, not of the data.
    """
    logits, feats = [], []
    asserted = 0
    for batch in loader:
        xb = batch[0].to(device, non_blocking=True)
        f = model.features(xb)
        lg = model.linear(f)
        if assert_first_batch and asserted == 0:
            assert_raw_logits(model, xb, lg)
            asserted = 1
        logits.append(lg.float().cpu().numpy())
        if want_feats:
            feats.append(f.float().cpu().numpy())
    return (np.concatenate(logits),
            np.concatenate(feats) if want_feats else None,
            asserted)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "ext_tier1.yaml"))
    p.add_argument("--runs-dir", default=os.path.join(ROOT, "results", "runs_ext"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "results", "forward_ext"))
    p.add_argument("--only", default="", help="comma-separated run_ids (default: all)")
    p.add_argument("--shard-tag", default="", help="override the derived summary tag")
    p.add_argument("--workers", type=int, default=6,
                   help="dataloader workers. Numerically inert here: the eval tensors are "
                        "materialized once in this process and iterated with shuffle=False, "
                        "so workers move batches but never produce a value.")
    p.add_argument("--consolidate-only", action="store_true",
                   help="merge existing shard summaries and exit; runs no inference")
    p.add_argument("--manifest-only", action="store_true",
                   help="write MANIFEST.sha256 over out-dir and exit; runs no inference")
    a = p.parse_args(argv)

    if a.consolidate_only:
        out = consolidate(a.out_dir)
        print(f"[fwd] consolidated {out['n_shards']} shard(s), {out['checkpoints']} "
              f"checkpoints over {out['runs']} runs; worst deviation "
              f"{out['max_abs_deviation_vs_log']:.3e}; SOP assertions "
              f"{out['sop_raw_logit_assertions']}/{out['sop_assertion_expectation']}")
        return 0
    if a.manifest_only:
        m = write_manifest(a.out_dir)
        print(f"[fwd] MANIFEST.sha256: {m['files']} files, {m['bytes']/1024**3:.1f} GiB")
        print(f"[fwd] manifest sha256: {m['manifest_sha256']}")
        return 0

    stamp = code_stamp()                    # stamped once, into every artifact written
    cfg = load_config(a.config)
    ecfg = cfg["eval"]
    batch = int(ecfg["eval_batch_size"])    # pinned at 512 by the sealed config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_ids = sorted(d for d in os.listdir(a.runs_dir)
                     if os.path.isfile(os.path.join(a.runs_dir, d, "TERMINAL.json")))
    if a.only:
        wanted = {x.strip() for x in a.only.split(",") if x.strip()}
        missing = wanted - set(run_ids)
        if missing:
            raise SystemExit(f"--only names runs with no TERMINAL.json: {sorted(missing)}")
        run_ids = [r for r in run_ids if r in wanted]

    shard_tag = a.shard_tag or shard_tag_for(run_ids)
    t_start = time.time()
    n_done = n_ck = n_assert = 0
    worst_dev = 0.0
    summary: List[dict] = []

    for run_id in run_ids:
        run_dir = os.path.join(a.runs_dir, run_id)
        lines = open(os.path.join(run_dir, "metrics.jsonl")).readlines()
        meta = json.loads(lines[0])
        logged = {int(json.loads(l)["epoch"]): json.loads(l) for l in lines[1:]}
        dataset, seed, learner = meta["dataset"], int(meta["seed"]), meta["learner"]
        split_name = meta["split"]
        n_classes = D.N_CLASSES[dataset]
        _seed_all(seed)

        # sealed real-noise labels, re-verified on load (sha256, rate, mask definition)
        real = load_split(split_name, os.path.join(ROOT, cfg["mask_dir_real"]),
                          cfg["real_noise_splits"])
        clean = D.clean_labels(dataset, cfg["data_root"], train=True)
        if not np.array_equal(np.asarray(clean, np.int64), real.clean):
            raise SystemExit(f"{run_id}: sealed clean labels for {split_name!r} disagree "
                             f"with the {dataset} training targets")
        if real.npz_sha256 != meta["noise_provenance"]["npz_sha256"]:
            raise SystemExit(f"{run_id}: split sha256 {real.npz_sha256} != the "
                             f"{meta['noise_provenance']['npz_sha256']} the run trained on")
        noisy = real.noisy

        tfm = D.eval_transform(dataset)
        assert_eval_transform_deterministic(tfm)
        train_ds = D.ArrayImageDataset(
            D._tv_cifar(dataset, cfg["data_root"], train=True).data,
            np.asarray(noisy), tfm)                             # EVAL transform: no aug
        test_ds = D.make_clean_test_dataset(dataset, cfg["data_root"])
        pools = load_ood_pools(os.path.join(ROOT, "results", "ood_pools"), dataset)

        def loader(ds):
            x, y = D.materialize_eval_tensors(ds)
            return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False,
                              num_workers=a.workers)

        loaders = {"train": loader(train_ds), "test": loader(test_ds)}
        for name, arr in pools.pools.items():
            loaders[name] = loader(D.ArrayImageDataset(
                arr, np.zeros(len(arr), np.int64), tfm))

        out_run = os.path.join(a.out_dir, run_id)
        os.makedirs(out_run, exist_ok=True)
        np.savez(os.path.join(out_run, "labels.npz"),
                 train_noisy=np.asarray(noisy, np.int64),
                 train_clean=np.asarray(real.clean, np.int64),
                 test=np.asarray(test_ds.labels, np.int64))

        model = build_model(cfg["model"]["arch"], n_classes).to(device)
        ckpts = sorted(f for f in os.listdir(run_dir) if f.startswith("checkpoint_ep"))
        for ck_name in ckpts:
            ck_path = os.path.join(run_dir, ck_name)
            epoch = int(ck_name[len("checkpoint_ep"):-len(".pt")])
            out_ep = os.path.join(out_run, f"ep{epoch:03d}")
            os.makedirs(out_ep, exist_ok=True)

            state = torch.load(ck_path, map_location=device)
            assert state["epoch"] == epoch, f"{ck_name} holds epoch {state['epoch']}"
            model.load_state_dict(state["model"])
            model.eval()

            t0 = time.time()
            lg_train, ft_train, asserted = infer(
                model, loaders["train"], device, want_feats=True,
                assert_first_batch=(learner == "sop"))
            np.save(os.path.join(out_ep, "logits_train.npy"), lg_train.astype(np.float32))
            np.save(os.path.join(out_ep, "feats_train.npy"), ft_train.astype(np.float32))
            lg_test, _f, _a = infer(model, loaders["test"], device, want_feats=False)
            np.save(os.path.join(out_ep, "logits_test.npy"), lg_test.astype(np.float32))
            pool_logits = {}
            for name in pools.pools:
                lg, _f, _a = infer(model, loaders[name], device, want_feats=False)
                pool_logits[name] = lg
                np.save(os.path.join(out_ep, f"logits_{name}.npy"), lg.astype(np.float32))
            secs = time.time() - t0
            n_assert += asserted

            # integrity: recompute the logged metrics from these logits. Same computation
            # path as the training-time evaluation, so the tolerance is exactly 0 (R4
            # trajectory-identity class) and a non-zero value aborts the shard.
            y = np.asarray(test_ds.labels)
            pred = lg_test.argmax(1)
            r_id = float((pred != y).mean())
            correct = np.zeros(n_classes); total = np.zeros(n_classes)
            np.add.at(total, y, 1.0); np.add.at(correct, y[pred == y], 1.0)
            pce = per_class_error(correct, total)
            r_tail = float(r_tail_dynamic(pce, ecfg["tail_frac"]))
            id_energy = energy_score(lg_test, ecfg["ood_energy_T"])
            sem = [1.0 - auroc(id_energy, energy_score(pool_logits[p], ecfg["ood_energy_T"]))
                   for p in ecfg["ood_semantic_pools"] if p in pool_logits]
            r_ood = float(np.mean(sem))
            lg_rec = logged[epoch]
            devs = {"R_ID": abs(r_id - lg_rec["R_ID"]),
                    "R_tail_dynamic": abs(r_tail - lg_rec["R_tail_dynamic"]),
                    "R_OOD_primary_energy_semantic":
                        abs(r_ood - lg_rec["R_OOD_primary_energy_semantic"])}
            dev = max(devs.values())
            worst_dev = max(worst_dev, dev)

            _atomic_json(os.path.join(out_ep, "meta.json"), dict(
                run_id=run_id, dataset=dataset, split=split_name, learner=learner,
                seed=seed, epoch=epoch, checkpoint_sha256=_sha256(ck_path),
                n_classes=n_classes, n_train=int(len(noisy)), n_test=int(len(y)),
                pools={k: int(len(v)) for k, v in pools.pools.items()},
                noise_provenance=dict(kind="real", split=split_name,
                                      npz_sha256=real.npz_sha256,
                                      measured_noise_rate_pct=real.measured_noise_rate_pct),
                batch_size=batch, dataloader_workers=a.workers, precision="fp32",
                torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), gpu_uuid=gpu_uuid(),
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                deterministic_algorithms=True, eval_transform=EVAL_TRANSFORM_DESC,
                reference_frame=REFERENCE_FRAME, code_stamp=stamp,
                sop_raw_logit_assertions=asserted,
                integrity_class="trajectory-identity (tolerance exactly 0)",
                recomputed=dict(R_ID=r_id, R_tail_dynamic=r_tail,
                                R_OOD_primary_energy_semantic=r_ood),
                logged={k: lg_rec[k] for k in devs},
                deviation_vs_log=devs, max_abs_deviation_vs_log=dev,
                seconds=round(secs, 2)))

            n_ck += 1
            print(f"[fwd] {run_id} ep{epoch:03d}  {secs:5.1f}s  dev {dev:.3e}"
                  f"{'  sop-assert ok' if asserted else ''}", flush=True)
            if dev != 0.0:
                _atomic_json(os.path.join(a.out_dir,
                                          f"ABORT_shard_{shard_tag}.json"),
                             dict(run_id=run_id, epoch=epoch, deviation=devs,
                                  max_abs_deviation_vs_log=dev, code_stamp=stamp))
                raise SystemExit(
                    f"INTEGRITY ABORT: {run_id} ep{epoch:03d} deviates from the log by "
                    f"{dev:.3e}; the tolerance for this check is exactly 0 "
                    f"(trajectory-identity class). Outputs so far are left in place for "
                    f"inspection; no consumer may read them until this is resolved.")

        n_done += 1
        summary.append(dict(run_id=run_id, split=split_name, learner=learner,
                            checkpoints=len(ckpts)))
        print(f"[fwd] RUN DONE {run_id} ({n_done}/{len(run_ids)})", flush=True)

    total = time.time() - t_start
    shard_path = os.path.join(a.out_dir, f"forward_summary_shard_{shard_tag}.json")
    _atomic_json(shard_path, dict(
        shard_tag=shard_tag, runs=len(run_ids), run_ids=run_ids, checkpoints=n_ck,
        code_stamp=stamp, max_abs_deviation_vs_log=worst_dev,
        integrity_class="trajectory-identity (tolerance exactly 0)",
        sop_raw_logit_assertions=n_assert,
        batch_size=batch, dataloader_workers=a.workers, precision="fp32",
        reference_frame=REFERENCE_FRAME,
        wall_clock_seconds=round(total, 1), per_run=summary))
    print(f"[fwd] {n_ck} checkpoints over {len(run_ids)} runs in {total/3600:.2f}h; "
          f"worst deviation {worst_dev:.3e}; SOP assertions {n_assert} -> {shard_path}")
    consolidate(a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
