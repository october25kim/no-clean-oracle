"""Refuse to launch when data/ holds files the sealed manifest does not account for.

The canonical data tree must contain only files whose provenance is recorded. A foreign
file appearing there is not automatically a problem -- it may be an unrelated download --
but a training campaign must not start without someone having looked at it, because the
audit's whole claim structure rests on knowing exactly which bytes trained the models.

This never deletes anything. It reports, and exits non-zero unless the caller passes
--allow-unmanifested, which is the operator saying "I have looked, and these files are
not inputs to this run".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MANIFEST = os.path.join(ROOT, "results", "cifar_n_masks", "MANIFEST.json")

# Filesystem noise that is never a data input and never worth flagging.
IGNORED_BASENAMES = {".DS_Store"}
IGNORED_PREFIXES = ("._",)          # macOS resource forks


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifested() -> Dict[str, str]:
    """{basename: sha256} for every file the seal accounts for."""
    with open(MANIFEST) as fh:
        m = json.load(fh)
    return dict(m.get("source_file_sha256", {}))


def scan(data_dir: str = DATA) -> Tuple[List[str], List[str], List[str]]:
    """Return (ok, unmanifested, hash_mismatch) as repo-relative paths."""
    known = manifested()
    ok, unknown, mismatch = [], [], []
    for dirpath, _dirnames, filenames in os.walk(data_dir):
        for name in sorted(filenames):
            if name in IGNORED_BASENAMES or name.startswith(IGNORED_PREFIXES):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, ROOT)
            if name in known:
                if _sha256(path) == known[name]:
                    ok.append(rel)
                else:
                    mismatch.append(rel)
            else:
                unknown.append(rel)
    return ok, unknown, mismatch


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA)
    p.add_argument("--allow-unmanifested", action="store_true",
                   help="proceed despite files data/ that the seal does not account for")
    a = p.parse_args(argv)

    if not os.path.isdir(a.data_dir):
        print(f"[data-hygiene] {a.data_dir} does not exist; nothing to check")
        return 0
    ok, unknown, mismatch = scan(a.data_dir)
    print(f"[data-hygiene] {len(ok)} manifested file(s) verified by sha256")

    if mismatch:
        print(f"[data-hygiene] {len(mismatch)} MANIFESTED FILE(S) CHANGED — the seal no "
              f"longer describes what is on disk:")
        for r in mismatch:
            print(f"    {r}")
        print("[data-hygiene] refusing; --allow-unmanifested does NOT override this.")
        return 2                       # a changed sealed input is never waivable

    if unknown:
        print(f"[data-hygiene] {len(unknown)} unmanifested file(s) in {a.data_dir}:")
        for r in unknown:
            size = os.path.getsize(os.path.join(ROOT, r))
            print(f"    {r}  ({size:,} bytes)")
        if not a.allow_unmanifested:
            print("[data-hygiene] refusing to launch. Nothing was deleted. Either remove "
                  "them yourself, add them to a seal, or re-run with "
                  "--allow-unmanifested to record that they are not inputs.")
            return 1
        print("[data-hygiene] proceeding under --allow-unmanifested; the files above are "
              "declared not to be inputs to this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
