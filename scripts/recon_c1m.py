"""[EXPLORATORY-UNVERIFIED-PROVENANCE] Clothing1M reconnaissance: train + forward + battery.

NOT Tier 2. Tier 2 stays gated on officially obtained data. Nothing this script writes may
enter a registered artifact, a paper claim, or be pooled with a registered result. Every
output file carries the classification tag, and everything lands under
``results/exploratory_c1m/``.

The registered launch path is deliberately not used, and `check_data_hygiene.py` is not
invoked. That guard exists to stop a *registered* campaign from starting on bytes nobody
has accounted for; here the inputs are known to be unaccounted-for — that is the whole
premise — and declaring them "not inputs" via ``--allow-unmanifested`` would be a false
statement, since they are precisely the inputs. Instead this script verifies each archive's
sha256 against ``PROVENANCE.md`` at load time and refuses to run on a mismatch, which is
the honest form of the same check.

Design pins, fixed here before any run and echoed into every output:

* backbone ResNet-50, torchvision ImageNet weights, version recorded;
* images are used at their native 64x64 — the bundle is already downsampled, and
  upsampling 1M images to the 224 px that standard Clothing1M practice uses would multiply
  compute by roughly an order of magnitude for a reconnaissance. Recorded as a deviation;
* train transform: pad-4 random crop + horizontal flip, ImageNet normalisation;
  eval transform: normalisation only, no randomness;
* SGD lr 0.01, momentum 0.9, weight decay 1e-3, batch 256, cosine to 0 over 20 epochs;
* {CE, ELR} x seeds {0, 1} = 4 runs, checkpoint every epoch -> a 20-point grid;
* ID and WC axes from the clean test set; WC = bottom round(0.30*14) = 4 classes by the
  reference run's final-epoch per-class accuracy, the protocol convention;
* NA selector input is a noisy split drawn from the training set, since the bundle has no
  clean train or val. OOD axis is NOT computed: the registered semantic-pool convention
  does not transfer and no analog has been confirmed.

ELR hyperparameters carry a gap that is recorded rather than hidden. ``configs/base.yaml``
holds registered rows for cifar10 (lam 3.0, beta 0.7) and cifar100 (lam 7.0, beta 0.9) and
none for Clothing1M, and ``build_loss`` raises rather than defaulting silently — by design.
The ELR paper's own Clothing1M setting is cited here from memory as lam 3, beta 0.7 and is
flagged NEEDS-VERIFICATION. What makes that a defensible exploratory choice rather than an
invention is that it coincides exactly with the registered cifar10 row, so the value in use
is one the project already has on record; the claim needing verification is only that the
paper uses the same pair for Clothing1M.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import zipfile
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

TAG = "EXPLORATORY-UNVERIFIED-PROVENANCE"
BUNDLE = os.path.join(ROOT, "data", "exploratory", "clothing1m_kaggle")
OUT = os.path.join(ROOT, "results", "exploratory_c1m")
N_CLASSES = 14
EPOCHS = 20
BATCH = 256
LR, MOMENTUM, WD = 0.01, 0.9, 1e-3
NA_SPLIT_N, NA_SEED = 5000, 20260813        # registered NA convention, reused
WC_FRAC = 0.30
ELR_LAM, ELR_BETA = 3.0, 0.7                # NEEDS-VERIFICATION; == registered cifar10 row
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRAIN_NPY = os.path.join(OUT, "train_images.npy")   # memmap-able extraction, shared


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def expected_digests() -> Dict[str, str]:
    """Digests as recorded in PROVENANCE.md — the file is the contract."""
    txt = open(os.path.join(BUNDLE, "PROVENANCE.md")).read()
    out = {}
    for m in re.finditer(r"\|\s*`([^`]+\.npz)`\s*\|[^|]*\|\s*`([0-9a-f]{64})`", txt):
        out[m.group(1)] = m.group(2)
    if len(out) != 2:
        raise SystemExit("PROVENANCE.md does not carry two npz digests")
    return out


def load_npz(name: str, verify: bool = True):
    path = os.path.join(BUNDLE, name)
    if verify:
        want = expected_digests()[name]
        got = sha256(path)
        if got != want:
            raise SystemExit(f"{name}: sha256 {got} != PROVENANCE.md {want}")
        print(f"[recon] {name} sha256 verified against PROVENANCE.md", flush=True)
    z = zipfile.ZipFile(path)
    with z.open("arr_0.npy") as f:
        x = np.lib.format.read_array(f)
    with z.open("arr_1.npy") as f:
        y = np.lib.format.read_array(f)
    return x, y


def load_labels(name: str) -> np.ndarray:
    """Digest-verified label read that does NOT materialise the image array.

    The training path memmaps the images from a one-time extraction, so pulling the 11.4
    GiB array through ``load_npz`` just to reach the labels beside it would cost minutes
    and gigabytes per run for nothing.
    """
    path = os.path.join(BUNDLE, name)
    want = expected_digests()[name]
    got = sha256(path)
    if got != want:
        raise SystemExit(f"{name}: sha256 {got} != PROVENANCE.md {want}")
    print(f"[recon] {name} sha256 verified against PROVENANCE.md", flush=True)
    with zipfile.ZipFile(path).open("arr_1.npy") as f:
        return np.lib.format.read_array(f)


def header_only(name: str):
    """Shapes/dtypes without materialising the arrays — used by --inspect."""
    z = zipfile.ZipFile(os.path.join(BUNDLE, name))
    out = {}
    for n in z.namelist():
        with z.open(n) as f:
            ver = np.lib.format.read_magic(f)
            shape, _fo, dt = np.lib.format._read_array_header(f, ver)
            out[n] = (shape, str(dt))
    return out


def na_split(noisy: np.ndarray, n_total: int = NA_SPLIT_N, seed: int = NA_SEED) -> np.ndarray:
    """Stratified-by-noisy-label draw, the registered NA convention (c1_noisyval pin)."""
    per = n_total // N_CLASSES
    rem = n_total - per * N_CLASSES
    rng = np.random.default_rng(seed)
    picked = []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(noisy == c)
        take = per + (1 if c < rem else 0)
        picked.append(rng.permutation(idx)[:take])
    return np.sort(np.concatenate(picked))


def inspect() -> int:
    """CPU-only preflight: digests, shapes, class balance, WC size, NA split, cost model."""
    print(f"[recon] [{TAG}] preflight — no GPU, no training")
    d = expected_digests()
    for name in ("clothing1m.npz", "clothing10k_test.npz"):
        got = sha256(os.path.join(BUNDLE, name))
        print(f"  {name:24s} sha256 {'OK' if got == d[name] else 'MISMATCH'}  {got[:16]}…")
        for k, (shape, dt) in header_only(name).items():
            print(f"      {k:12s} {shape} {dt}")
    _, ytr = load_npz("clothing1m.npz", verify=False) if False else (None, None)

    z = zipfile.ZipFile(os.path.join(BUNDLE, "clothing1m.npz"))
    with z.open("arr_1.npy") as f:
        ytr = np.lib.format.read_array(f)
    z = zipfile.ZipFile(os.path.join(BUNDLE, "clothing10k_test.npz"))
    with z.open("arr_1.npy") as f:
        yte = np.lib.format.read_array(f)

    print(f"  train labels {ytr.size:,} over {np.unique(ytr).size} classes")
    print(f"  test  labels {yte.size:,} over {np.unique(yte).size} classes")
    k = max(1, int(round(WC_FRAC * N_CLASSES)))
    cnt = np.bincount(yte, minlength=N_CLASSES)
    print(f"  WC set size k = max(1, round({WC_FRAC}*{N_CLASSES})) = {k}")
    print(f"  smallest {k} test classes by count: {sorted(cnt)[:k]} "
          f"(WC risk is a mean over whichever {k} the reference run ranks last)")
    idx = na_split(ytr)
    print(f"  NA split: {idx.size:,} indices, stratified by noisy label, seed {NA_SEED}")

    px = 64 * 64 * 3
    print(f"  train tensor if held in RAM: {ytr.size * px / 1024**3:.1f} GiB uint8")
    print(f"  checkpoints: {EPOCHS}/run x 4 runs = {EPOCHS * 4}; ResNet-50 ~ 94 MiB each "
          f"-> ~{EPOCHS * 4 * 94 / 1024:.1f} GiB")
    print(f"  forward pass: {EPOCHS * 4} checkpoints x {yte.size:,} test images"
          f" + {idx.size:,} NA images")
    return 0


# --------------------------------------------------------------------------- training

def extract_train_images() -> str:
    """Extract arr_0 once to a plain .npy so both concurrent runs can memmap one copy.

    Loading the 11.4 GiB array into each process would cost twice the RAM and read it
    twice; a shared memmap lets the page cache serve both waves.

    Both runs in a wave start together, so the extraction has to be safe against two
    processes reaching it at once. The first attempt used a single shared ``.tmp`` path and
    was killed sixty seconds in, when both containers reported extracting simultaneously —
    they would have interleaved writes into the same file and the atomic rename would then
    have published whichever garbage won. Each writer now uses its own pid-suffixed temp
    file, and an O_EXCL lock elects a single extractor while the others wait for the result
    rather than duplicating an 11.4 GiB read.
    """
    if os.path.isfile(TRAIN_NPY):
        return TRAIN_NPY
    os.makedirs(OUT, exist_ok=True)
    lock = TRAIN_NPY + ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        print("[recon] another process is extracting; waiting ...", flush=True)
        for _ in range(3600):
            if os.path.isfile(TRAIN_NPY):
                return TRAIN_NPY
            time.sleep(5)
        raise SystemExit("timed out waiting for the shared extraction")

    print("[recon] extracting train images to a memmap-able .npy (once) ...", flush=True)
    tmp = f"{TRAIN_NPY}.tmp.{os.getpid()}"
    try:
        z = zipfile.ZipFile(os.path.join(BUNDLE, "clothing1m.npz"))
        with z.open("arr_0.npy") as src, open(tmp, "wb") as dst:
            while True:
                b = src.read(1 << 24)
                if not b:
                    break
                dst.write(b)
        os.replace(tmp, TRAIN_NPY)
    finally:
        for p in (tmp, lock):
            if os.path.exists(p) and p != TRAIN_NPY:
                try:
                    os.remove(p)
                except OSError:
                    pass
    return TRAIN_NPY


def train_one(learner: str, seed: int, batch: int, gpu: int | None = None) -> dict:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import torchvision

    sys.path.insert(0, os.path.join(ROOT, "src"))
    from provenance import code_stamp
    from train.elr import ELRLoss

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    run_id = f"c1m_{learner}_seed{seed}"
    run_dir = os.path.join(OUT, run_id)
    os.makedirs(run_dir, exist_ok=True)
    # A bare "cuda" is device 0 of whatever the container can see, which under
    # --gpus "device=<uuid>" is the single pinned GPU -- so wave 1's bare "cuda" was correct
    # and its empty CUDA_VISIBLE_DEVICES was expected. That combination nonetheless reads
    # exactly like an unpinned run, and it was misread as one (defect ledger D-9, withdrawn):
    # the artifact could not say which device it used, so the question went to the launcher
    # instead, and the launcher was read wrong. This is instrumentation, not a fix. Pin when
    # asked, fail closed on a bad index, and write the resolved device into meta so the
    # artifact answers the placement question itself.
    if gpu is not None:
        n_dev = torch.cuda.device_count()
        if not 0 <= gpu < n_dev:
            raise SystemExit(f"--gpu {gpu} out of range; {n_dev} CUDA device(s) visible")
        torch.cuda.set_device(gpu)
        dev = f"cuda:{gpu}"
    else:
        dev = "cuda"
    dev_index = torch.cuda.current_device()
    dev_name = torch.cuda.get_device_name(dev_index)

    extract_train_images()
    x = np.load(TRAIN_NPY, mmap_mode="r")
    ytr = np.asarray(load_labels("clothing1m.npz"), np.int64)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    class TrainDS(Dataset):
        def __len__(self): return len(ytr)
        def __getitem__(self, i):
            img = torch.from_numpy(np.ascontiguousarray(x[i])).permute(2, 0, 1).float() / 255
            img = F.pad(img.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
            t, l = random.randint(0, 8), random.randint(0, 8)
            img = img[:, t:t + 64, l:l + 64]
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])
            return (img - mean) / std, int(ytr[i]), i

    loader = DataLoader(TrainDS(), batch_size=batch, shuffle=True, num_workers=6,
                        pin_memory=True, drop_last=False, persistent_workers=True)

    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet50(weights=w)
    net.fc = torch.nn.Linear(net.fc.in_features, N_CLASSES)
    net = net.to(dev)

    loss_fn = None if learner == "ce" else ELRLoss(len(ytr), N_CLASSES, ELR_LAM, ELR_BETA)
    if loss_fn is not None and hasattr(loss_fn, "to"):
        loss_fn.to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    meta = dict(classification=TAG, run_id=run_id, learner=learner, seed=seed,
                design_pin_commit="b333df8", backbone="resnet50", weights=str(w),
                weights_url=w.url, input="native 64x64, no upsampling",
                train_transform="reflect-pad 4 + random crop 64 + hflip + ImageNet norm",
                eval_transform="ImageNet norm only (deterministic)",
                optimizer=dict(sgd=dict(lr=LR, momentum=MOMENTUM, weight_decay=WD),
                               batch=batch, schedule="cosine to 0", epochs=EPOCHS),
                elr=dict(lam=ELR_LAM, beta=ELR_BETA,
                         note="NEEDS-VERIFICATION; equals the registered cifar10 row")
                if learner == "elr" else None,
                n_train=int(len(ytr)), n_classes=N_CLASSES,
                torch=torch.__version__, cuda=torch.version.cuda,
                cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                device=dev, device_index=dev_index, device_name=dev_name,
                code_stamp=code_stamp())
    with open(os.path.join(run_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        tot = n = 0
        for xb, yb, idx in loader:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            out = net(xb)
            loss = (F.cross_entropy(out, yb) if learner == "ce"
                    else loss_fn(out, yb, idx.to(dev)))
            loss.backward()
            opt.step()
            tot += float(loss) * xb.size(0); n += xb.size(0)
        sched.step()
        torch.save(dict(epoch=ep, model=net.state_dict()),
                   os.path.join(run_dir, f"checkpoint_ep{ep:03d}.pt"))
        rec = dict(epoch=ep, lr=float(sched.get_last_lr()[0]), train_loss=tot / max(n, 1),
                   seconds=round(time.time() - t0, 1))
        with open(os.path.join(run_dir, "metrics.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[recon] {run_id} ep{ep:03d} loss {rec['train_loss']:.4f} "
              f"{rec['seconds']:.0f}s", flush=True)

    # meta already carries `classification`, so splatting it beside an explicit
    # classification= raised TypeError -- after 20 epochs, with every checkpoint on disk.
    # Worse, `with open(...)` had already created the file, so a 0-byte TERMINAL.json sat
    # there looking present to any existence check. Build the payload FIRST, open only to
    # write it, and never re-pass a key meta owns.
    payload = dict(meta)
    payload.update(status="completed", epochs=EPOCHS,
                   wall_seconds=round(time.time() - t0, 1))
    tmp = os.path.join(run_dir, "TERMINAL.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, os.path.join(run_dir, "TERMINAL.json"))
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", action="store_true",
                   help="CPU-only preflight; verifies digests and prints the design")
    p.add_argument("--learner", choices=["ce", "elr"])
    p.add_argument("--seed", type=int)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--gpu", type=int, default=None,
                   help="CUDA device index to pin; omit only to reproduce wave 1's default")
    a = p.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if a.inspect:
        return inspect()
    if a.learner is None or a.seed is None:
        raise SystemExit("need --learner and --seed (or --inspect)")
    train_one(a.learner, a.seed, a.batch, a.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
