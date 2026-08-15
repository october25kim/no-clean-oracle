"""Approved B2 forward pass: 15 runs x 24 checkpoints -> logits + embeddings.

Inference only. Loads each retained checkpoint, runs the pinned evaluation sets through
it once, and writes the arrays every downstream consumer needs (see
docs/forward_pass_schema.md). No training, no checkpoint is modified, and ``latest.pt``
is never read -- it is a resume artifact, not part of the selection grid.

Every checkpoint's metrics are recomputed from its own logits and compared against what
the B2 run logged at that epoch. A non-zero deviation means the pass is not reproducing
the training-time evaluation, which would invalidate the outputs, so it is recorded per
checkpoint and summarised at the end.
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
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from config import load_config                          # noqa: E402
from data import datasets as D                          # noqa: E402
from data.noise import NoiseConfig, load_or_make_noisy_labels   # noqa: E402
from data.ood_pools import load_ood_pools               # noqa: E402
from eval.ood import auroc, energy_score, msp_score     # noqa: E402
from eval.tail import per_class_error, r_tail_dynamic   # noqa: E402
from models.preact_resnet import build_model            # noqa: E402
from provenance import code_stamp                       # noqa: E402
from run_single import gpu_uuid                         # noqa: E402


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
    """Write via a per-process temp file, then rename.

    Concurrent shards must never leave a half-written summary, and must never race on
    the same final name -- the first forward pass had four shards writing one
    ``forward_summary.json``, so only the last shard's 96 checkpoints survived and the
    file silently described a quarter of the sweep.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def shard_tag_for(run_ids: List[str]) -> str:
    """Deterministic tag from the runs a shard owns: same shard -> same filename."""
    return hashlib.sha256("|".join(sorted(run_ids)).encode()).hexdigest()[:10]


def consolidate(out_dir: str) -> dict:
    """Merge every shard summary present into one consolidated file.

    Safe to call from each shard as it finishes: it only reads whatever shards have
    landed, and writes atomically, so the last caller leaves the complete picture. Run
    it again with --consolidate-only if a shard is added or re-run later.
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
    out = dict(
        consolidated_from=[s["shard_tag"] for s in shards],
        n_shards=len(shards), runs=len(runs), checkpoints=len(metas),
        checkpoints_per_run={r: sum(1 for m in metas if m["run_id"] == r) for r in runs},
        max_abs_deviation_vs_log=max(devs),
        n_checkpoints_with_nonzero_deviation=sum(1 for d in devs if d != 0.0),
        batch_size=sorted({m["batch_size"] for m in metas}),
        precision=sorted({m["precision"] for m in metas}),
        reference_frame=sorted({m["reference_frame"] for m in metas}),
        gpu_uuids=sorted({m["gpu_uuid"] for m in metas if m.get("gpu_uuid")}),
        torch=sorted({m["torch"] for m in metas}),
        cuda=sorted({m["cuda"] for m in metas}),
        cudnn=sorted({m["cudnn"] for m in metas}),
        cublas_workspace_config=sorted({m["cublas_workspace_config"] for m in metas}),
        eval_transform=sorted({m["eval_transform"] for m in metas}),
        code_stamp_present=any("code_stamp" in m for m in metas),
        code_stamp_note=("absent in outputs from any process launched before commit "
                         "2f3d9db, which added the stamp; such outputs are attributed "
                         "by blob-hash chain instead and are never retro-patched"),
        compute_seconds_total=round(sum(secs), 1),
        per_checkpoint_seconds=dict(mean=round(sum(secs) / max(len(secs), 1), 2),
                                    min=round(min(secs), 2), max=round(max(secs), 2)),
        shard_wall_clock_seconds={s["shard_tag"]: s["wall_clock_seconds"] for s in shards},
    )
    _atomic_json(os.path.join(out_dir, "forward_summary_consolidated.json"), out)
    return out


@torch.no_grad()
def infer(model, loader, device: str, want_feats: bool):
    logits, feats = [], []
    for batch in loader:
        xb = batch[0].to(device, non_blocking=True)
        f = model.features(xb)
        logits.append(model.linear(f).float().cpu().numpy())
        if want_feats:
            feats.append(f.float().cpu().numpy())
    return (np.concatenate(logits),
            np.concatenate(feats) if want_feats else None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "b2.yaml"))
    p.add_argument("--runs-dir", default=os.path.join(ROOT, "results", "runs_b2"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "results", "forward_b2"))
    p.add_argument("--only", default="", help="comma-separated run_ids (default: all)")
    p.add_argument("--shard-tag", default="",
                   help="override the derived per-shard summary tag")
    p.add_argument("--consolidate-only", action="store_true",
                   help="merge existing shard summaries and exit; runs no inference")
    a = p.parse_args(argv)

    if a.consolidate_only:
        out = consolidate(a.out_dir)
        print(f"[fwd] consolidated {out['n_shards']} shard(s), "
              f"{out['checkpoints']} checkpoints over {out['runs']} runs; "
              f"worst deviation {out['max_abs_deviation_vs_log']:.3e}")
        return 0

    # stamped once, at process start, into every artifact this process writes
    stamp = code_stamp()

    cfg = load_config(a.config)
    ecfg = cfg["eval"]
    batch = int(ecfg["eval_batch_size"])          # pinned at 512 by the sealed config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_ids = sorted(d for d in os.listdir(a.runs_dir)
                     if os.path.isfile(os.path.join(a.runs_dir, d, "TERMINAL.json")))
    if a.only:
        wanted = {x.strip() for x in a.only.split(",") if x.strip()}
        run_ids = [r for r in run_ids if r in wanted]

    shard_tag = a.shard_tag or shard_tag_for(run_ids)
    t_start = time.time()
    n_done = n_ck = 0
    worst_dev = 0.0
    summary: List[dict] = []

    for run_id in run_ids:
        run_dir = os.path.join(a.runs_dir, run_id)
        meta = json.loads(open(os.path.join(run_dir, "metrics.jsonl")).readline())
        logged = {int(json.loads(l)["epoch"]): json.loads(l)
                  for l in open(os.path.join(run_dir, "metrics.jsonl")).readlines()[1:]}
        dataset, seed = meta["dataset"], int(meta["seed"])
        n_classes = D.N_CLASSES[dataset]
        _seed_all(seed)

        # the same frozen inputs the run trained and was evaluated on
        clean = D.clean_labels(dataset, cfg["data_root"], train=True)
        noisy = load_or_make_noisy_labels(
            os.path.join(ROOT, cfg["mask_dir"]),
            NoiseConfig(dataset, meta["noise_type"], float(meta["eta"]), seed), clean)
        train_ds = D.ArrayImageDataset(
            D._tv_cifar(dataset, cfg["data_root"], train=True).data,
            np.asarray(noisy), D.eval_transform(dataset))       # EVAL transform: no aug
        test_ds = D.make_clean_test_dataset(dataset, cfg["data_root"])
        pools = load_ood_pools(os.path.join(ROOT, "results", "ood_pools"), dataset)

        def loader(ds):
            x, y = D.materialize_eval_tensors(ds)
            return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False,
                              num_workers=0)

        loaders = {"train": loader(train_ds), "test": loader(test_ds)}
        for name, arr in pools.pools.items():
            loaders[name] = loader(D.ArrayImageDataset(
                arr, np.zeros(len(arr), np.int64), D.eval_transform(dataset)))

        out_run = os.path.join(a.out_dir, run_id)
        os.makedirs(out_run, exist_ok=True)
        np.savez(os.path.join(out_run, "labels.npz"),
                 train_noisy=np.asarray(noisy, np.int64),
                 train_clean=np.asarray(clean, np.int64),
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
            lg_train, ft_train = infer(model, loaders["train"], device, want_feats=True)
            np.save(os.path.join(out_ep, "logits_train.npy"), lg_train.astype(np.float32))
            np.save(os.path.join(out_ep, "feats_train.npy"), ft_train.astype(np.float32))
            lg_test, _ = infer(model, loaders["test"], device, want_feats=False)
            np.save(os.path.join(out_ep, "logits_test.npy"), lg_test.astype(np.float32))
            pool_logits = {}
            for name in pools.pools:
                lg, _ = infer(model, loaders[name], device, want_feats=False)
                pool_logits[name] = lg
                np.save(os.path.join(out_ep, f"logits_{name}.npy"), lg.astype(np.float32))
            secs = time.time() - t0

            # integrity: recompute the logged metrics from these logits
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

            with open(os.path.join(out_ep, "meta.json"), "w") as fh:
                json.dump(dict(
                    run_id=run_id, dataset=dataset, learner=meta["learner"], seed=seed,
                    epoch=epoch, checkpoint_sha256=_sha256(ck_path),
                    n_classes=n_classes, n_train=int(len(noisy)), n_test=int(len(y)),
                    pools={k: int(len(v)) for k, v in pools.pools.items()},
                    batch_size=batch, precision="fp32", torch=torch.__version__,
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    gpu_uuid=gpu_uuid(),
                    cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                    deterministic_algorithms=True,
                    eval_transform="ToTensor+Normalize (no randomness)",
                    reference_frame="G1", code_stamp=stamp,
                    recomputed=dict(R_ID=r_id, R_tail_dynamic=r_tail,
                                    R_OOD_primary_energy_semantic=r_ood),
                    logged={k: lg_rec[k] for k in devs},
                    deviation_vs_log=devs, max_abs_deviation_vs_log=dev,
                    seconds=round(secs, 2)), fh, indent=2)

            n_ck += 1
            print(f"[fwd] {run_id} ep{epoch:03d}  {secs:5.1f}s  dev {dev:.3e}", flush=True)

        n_done += 1
        summary.append(dict(run_id=run_id, checkpoints=len(ckpts)))
        print(f"[fwd] RUN DONE {run_id} ({n_done}/{len(run_ids)})", flush=True)

    total = time.time() - t_start
    shard_path = os.path.join(a.out_dir, f"forward_summary_shard_{shard_tag}.json")
    _atomic_json(shard_path, dict(
        shard_tag=shard_tag, runs=len(run_ids), run_ids=run_ids, checkpoints=n_ck,
        code_stamp=stamp, max_abs_deviation_vs_log=worst_dev,
        batch_size=batch, precision="fp32", reference_frame="G1",
        wall_clock_seconds=round(total, 1), per_run=summary))
    print(f"[fwd] {n_ck} checkpoints over {len(run_ids)} runs in {total/3600:.2f}h; "
          f"worst metric deviation vs log = {worst_dev:.3e} -> {shard_path}")
    consolidate(a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
