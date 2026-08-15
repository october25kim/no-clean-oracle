"""One G1 run: (dataset x noise x learner x seed) -> per-epoch JSONL + checkpoints.

Single-GPU, deterministic, resumable. Identical whether launched serially or pinned
to a GPU by the sweep scheduler (which sets CUDA_VISIBLE_DEVICES and passes
--assigned-gpu for provenance only).
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import load_config                     # noqa: E402
from provenance import code_stamp                  # noqa: E402
from data import datasets as D                     # noqa: E402
from data.noise import NoiseConfig, build_transition_matrix, load_or_make_noisy_labels  # noqa: E402
from data.ood_pools import load_ood_pools          # noqa: E402
from data.real_noise import load_split            # noqa: E402
from eval.evaluate import evaluate_epoch, train_diagnostics  # noqa: E402
from models.preact_resnet import build_model       # noqa: E402
from train.trainer import Trainer, TrainConfig, build_loss  # noqa: E402


class _Indexed(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i


def gpu_uuid() -> Optional[str]:
    """Physical UUID of the GPU this run trains on -- provenance for co-located runs.

    nvidia-smi does NOT honour CUDA_VISIBLE_DEVICES, so its listing is indexed by
    hand. The sweep launcher sets CUDA_DEVICE_ORDER=PCI_BUS_ID so that index agrees
    with the one CUDA hands to torch; a single-GPU container has only index 0.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True, timeout=30)
    except Exception:
        return None
    by_index = {}
    for line in out.strip().splitlines():
        if "," not in line:
            continue
        i, u = line.split(",", 1)
        by_index[int(i.strip())] = u.strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return by_index.get(0)
    first = visible.split(",")[0].strip()
    return first if first.startswith("GPU-") else by_index.get(int(first))


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", default="")
    p.add_argument("--noise-type", default="")
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--split", default="",
                   help="sealed real-noise split name (exact match), e.g. c10n_worst; "
                        "mutually exclusive with --noise-type/--eta")
    p.add_argument("--learner", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--assigned-gpu", type=int, default=-1)
    p.add_argument("--co-tenant", default="", help="name of a co-located workload sharing this GPU")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs (smoke)")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_all(a.seed)

    data_root = cfg["data_root"]
    real = None
    if a.split:                       # real-noise (CIFAR-N) run
        if a.noise_type or a.eta is not None:
            raise SystemExit("--split is mutually exclusive with --noise-type/--eta")
        real = load_split(a.split, os.path.join(ROOT, cfg["mask_dir_real"]),
                          cfg["real_noise_splits"])
        dataset = real.dataset
        n_classes = D.N_CLASSES[dataset]
        clean = D.clean_labels(dataset, data_root, train=True)
        if not np.array_equal(np.asarray(clean, np.int64), real.clean):
            raise SystemExit(f"sealed clean labels for {a.split!r} disagree with the "
                             f"{dataset} training targets the loader will use")
        noisy = real.noisy
        eff_overall = real.measured_noise_rate_pct / 100.0
        run_id = f"{a.split}_{a.learner}_seed{a.seed}"
        noise_provenance = dict(kind="real", split=a.split,
                                npz_sha256=real.npz_sha256,
                                measured_noise_rate_pct=real.measured_noise_rate_pct)
    else:                             # synthetic noise, exactly as G1/B2
        if not (a.dataset and a.noise_type and a.eta is not None):
            raise SystemExit("synthetic runs need --dataset --noise-type --eta")
        dataset = a.dataset
        n_classes = D.N_CLASSES[dataset]
        clean = D.clean_labels(dataset, data_root, train=True)
        ncfg = NoiseConfig(dataset, a.noise_type, a.eta, a.seed)
        noisy = load_or_make_noisy_labels(os.path.join(ROOT, cfg["mask_dir"]), ncfg, clean)
        T = build_transition_matrix(dataset, a.noise_type, a.eta)
        eff_overall = float(1.0 - np.mean(np.diag(T)))
        run_id = f"{dataset}_{a.noise_type}{a.eta:g}_{a.learner}_seed{a.seed}"
        noise_provenance = dict(kind="synthetic", noise_type=a.noise_type, eta=a.eta)
    run_dir = os.path.join(ROOT, cfg["results_dir"], run_id)

    # Only the TRAIN loader uses worker processes: its augmentation RNG is seeded per
    # worker, so num_workers is part of the run's reproducibility contract and must be
    # identical across every run of the sweep. Eval sets are transformed once into RAM
    # (see materialize_eval_tensors) and iterated with num_workers=0.
    workers = int(os.environ.get("G1_DATALOADER_WORKERS", str(max(1, (os.cpu_count() or 4) // 4))))
    train_ds = _Indexed(D.make_train_dataset(dataset, data_root, noisy))
    g = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=workers, drop_last=False, generator=g, persistent_workers=False)
    ecfg = cfg["eval"]
    eval_bs = int(ecfg["eval_batch_size"])

    def _eval_loader(ds) -> DataLoader:
        x, y = D.materialize_eval_tensors(ds)
        return DataLoader(TensorDataset(x, y), batch_size=eval_bs, shuffle=False, num_workers=0)

    test_loader = _eval_loader(D.make_clean_test_dataset(dataset, data_root))

    # OOD pools (frozen artifact) -> cached eval tensors, resident for the whole run
    pools = load_ood_pools(os.path.join(ROOT, "results", "ood_pools"), dataset)
    ood_loaders = {name: _eval_loader(D.ArrayImageDataset(arr, np.zeros(len(arr), np.int64),
                                                          D.eval_transform(dataset)))
                   for name, arr in pools.pools.items()}

    def eval_fn(model):
        return evaluate_epoch(model, device, test_loader, n_classes, tail_static=None,
                              ood_pool_loaders=ood_loaders,
                              semantic_pools=ecfg["ood_semantic_pools"],
                              covariate_pool=ecfg["ood_covariate_pool"],
                              energy_T=ecfg["ood_energy_T"], tail_frac=ecfg["tail_frac"])

    # learner hyperparameters, with the per-dataset row folded in (ELR uses different
    # lambda/beta for CIFAR-10 and CIFAR-100); recorded verbatim in the metadata.
    lparams = dict(cfg["learners"].get(a.learner) or {})
    lparams.update((lparams.pop("per_dataset", None) or {}).get(dataset, {}))
    # per_split is keyed by the EXACT split name. Popped unconditionally so the raw
    # table never reaches the metadata, but only consulted for real-noise runs.
    per_split = lparams.pop("per_split", None) or {}
    if a.split:
        if per_split and a.split not in per_split:
            raise SystemExit(f"learner {a.learner!r} has a per_split table but no entry "
                             f"for {a.split!r}; entries are {sorted(per_split)}")
        lparams.update(per_split.get(a.split, {}))

    model = build_model(cfg["model"]["arch"], n_classes)
    loss_fn = build_loss(a.learner, len(clean), n_classes, lparams)
    if hasattr(loss_fn, "to"):
        loss_fn.to(device)
    epochs = a.epochs if a.epochs is not None else cfg["train"]["epochs"]
    tcfg = TrainConfig(epochs=epochs, lr=cfg["train"]["lr"],
                       momentum=cfg["train"]["momentum"], weight_decay=cfg["train"]["weight_decay"],
                       batch_size=cfg["train"]["batch_size"], lr_schedule=cfg["train"]["lr_schedule"],
                       save_every=cfg["checkpointing"]["save_every"])
    metadata = dict(run_id=run_id, dataset=dataset, noise_type=(a.noise_type or None),
                    eta=a.eta, split=(a.split or None), noise_provenance=noise_provenance,
                    learner=a.learner, learner_params=lparams, seed=a.seed,
                    assigned_gpu=a.assigned_gpu, gpu_uuid=gpu_uuid(),
                    co_tenant=(a.co_tenant or None), smoke=(a.epochs is not None),
                    epochs=epochs, effective_overall_noise=eff_overall, n_classes=n_classes,
                    tail_frac=ecfg["tail_frac"], train_dataloader_workers=workers,
                    eval_batch_size=eval_bs, ood_pools=list(pools.pools),
                    ood_provenance=pools.meta,
                    ood_subsample_seed=pools.meta.get("cifar_c_local", {}).get("subsample_seed"),
                    torch=torch.__version__, cuda=torch.version.cuda,
                    cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    code_stamp=code_stamp())
    trainer = Trainer(model, loss_fn, tcfg, device, run_dir, metadata, eval_fn, train_diagnostics)
    trainer.acquire_lock()          # nothing else may write this run_dir
    try:
        if a.resume:
            trainer.maybe_resume()
        trainer.train(train_loader, n_classes, np.asarray(noisy))
    finally:
        trainer.release_lock()
    print(f"[run_single] {run_id} DONE -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
