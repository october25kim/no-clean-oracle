"""Prove that every learner in a cell trained on the IDENTICAL corrupted labels.

``load_or_make_noisy_labels`` writes a cell's mask once and every later run loads it,
so "CE and ELR saw the same labels" reduces to "the mask file never changed after it
was first written". This records a manifest of sha256 + size + mtime the first time it
runs and verifies against it afterwards; a changed or missing mask is a hard failure.

  python scripts/verify_masks.py            # verify (writes the manifest if absent)
  python scripts/verify_masks.py --update   # re-baseline deliberately
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_DIR = os.path.join(ROOT, "results", "noise_masks")
MANIFEST = os.path.join(MASK_DIR, "MANIFEST.json")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def survey(mask_dir: str = MASK_DIR) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(mask_dir, "*.npz"))):
        d = np.load(path)
        out[os.path.basename(path)] = dict(
            sha256=_sha256(path), bytes=os.path.getsize(path),
            mtime=round(os.path.getmtime(path), 3), n=int(len(d["noisy"])),
            labels_sha256=hashlib.sha256(np.ascontiguousarray(d["noisy"]).tobytes()).hexdigest(),
            empirical_flip_rate=float((d["clean"] != d["noisy"]).mean()))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--update", action="store_true", help="overwrite the manifest baseline")
    a = p.parse_args(argv)

    current = survey()
    if not current:
        print(f"no masks under {MASK_DIR}")
        return 1

    if a.update or not os.path.isfile(MANIFEST):
        with open(MANIFEST, "w") as fh:
            json.dump(current, fh, indent=2, sort_keys=True)
        print(f"[verify_masks] wrote baseline for {len(current)} mask(s) -> {MANIFEST}")
        for name, rec in current.items():
            print(f"  {name}  {rec['sha256'][:16]}…  n={rec['n']}  flip={rec['empirical_flip_rate']:.4f}")
        return 0

    with open(MANIFEST) as fh:
        base = json.load(fh)
    changed = [n for n in base if n in current and current[n]["sha256"] != base[n]["sha256"]]
    missing = [n for n in base if n not in current]
    added = [n for n in current if n not in base]

    for name in sorted(current):
        rec = current[name]
        tag = "NEW " if name in added else ("CHANGED" if name in changed else "ok  ")
        print(f"  [{tag}] {name}  {rec['sha256'][:16]}…  n={rec['n']}  "
              f"flip={rec['empirical_flip_rate']:.4f}")
    for name in missing:
        print(f"  [GONE] {name}")

    if changed or missing:
        print(f"[verify_masks] FAIL — {len(changed)} changed, {len(missing)} missing. "
              f"Runs in those cells did NOT all train on the same labels.")
        return 1
    print(f"[verify_masks] PASS — {len(base)} baselined mask(s) unchanged"
          + (f", {len(added)} new mask(s) recorded on first use" if added else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
