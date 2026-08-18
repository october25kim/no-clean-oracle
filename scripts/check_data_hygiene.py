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
import random
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MANIFEST = os.path.join(ROOT, "results", "cifar_n_masks", "MANIFEST.json")
EXPLORATORY_MANIFEST = os.path.join(ROOT, "MANIFEST.exploratory.json")
EXPLORATORY_ROOT = os.path.join("data", "exploratory")
OFFICIAL_ROOT = os.path.join("data", "clothing1m_official")
OFFICIAL_CHECKSUMS = os.path.join(ROOT, OFFICIAL_ROOT, "CHECKSUMS.sha256")
SAMPLE_N = 64                      # deterministic spot-check when not running --deep

# Filesystem noise that is never a data input and never worth flagging.
IGNORED_BASENAMES = {".DS_Store"}
IGNORED_PREFIXES = ("._",)          # macOS resource forks

# A tree's own pin and provenance record are not inputs to anything; they describe the
# inputs. Listing them as unmanifested would make the guard demand that the manifest
# manifest itself.
SELF_DESCRIBING = {"CHECKSUMS.sha256", "PROVENANCE.md", "MANIFEST.json"}


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


def exploratory() -> Dict[str, str]:
    """{repo-relative path: sha256} for exploratory-class entries, keyed by PATH.

    Keyed by path rather than basename, unlike the seal: a seal accounts for named data
    files wherever they sit, whereas an exploratory entry is a statement about one
    specific file in one specific place, and should not silently cover a same-named file
    somewhere else.
    """
    if not os.path.isfile(EXPLORATORY_MANIFEST):
        return {}
    with open(EXPLORATORY_MANIFEST) as fh:
        m = json.load(fh)
    return {p: e["sha256"] for p, e in m.get("entries", {}).items()}


def official() -> Dict[str, str]:
    """{repo-relative path: sha256} from the official tree's own CHECKSUMS.sha256.

    The official release is pinned by its own manifest rather than by the CIFAR-N seal, so
    the guard has to read that file or else treat 1,072,417 legitimate inputs as foreign and
    refuse every launch -- which is exactly what it did on first contact.
    """
    if not os.path.isfile(OFFICIAL_CHECKSUMS):
        return {}
    out = {}
    with open(OFFICIAL_CHECKSUMS) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            digest, _, rel = line.partition("  ")
            if digest and rel:
                out[os.path.join(OFFICIAL_ROOT, rel)] = digest
    return out


def missing_from_disk(*manifests: Dict[str, str]) -> List[str]:
    """The manifest->disk direction: entries that describe files which are not there.

    The disk->manifest direction alone cannot see a deletion. When the exploratory tree was
    removed, every one of its three manifested files vanished and the guard reported a clean
    run, because it only ever walked what existed. A manifest that describes a world that is
    gone is not a weaker pin, it is a false one.
    """
    gone = []
    for m in manifests:
        for rel in m:
            if not os.path.exists(os.path.join(ROOT, rel)):
                gone.append(rel)
    return sorted(gone)


def scan(data_dir: str = DATA, deep: bool = False):
    """Return (ok, unmanifested, hash_mismatch, exploratory_ok) as repo-relative paths.

    Exploratory-class entries under ``data/exploratory/**`` are recognised so that their
    presence cannot hard-block a registered launch — exploration must never gate the
    registered track. They are still checked: a changed exploratory file fails exactly as
    a changed sealed file does, because the manifest's whole content is the claim that
    these bytes have not moved since they were recorded. What it does NOT claim is where
    they came from; that is the point of the classification.
    """
    known = manifested()
    expl = exploratory()
    offi = official()
    ok, unknown, mismatch, expl_ok = [], [], [], []
    off_ok = []
    rng = random.Random(20260818)
    sampled = set(rng.sample(sorted(offi), min(SAMPLE_N, len(offi)))) if offi else set()
    for dirpath, _dirnames, filenames in os.walk(data_dir):
        for name in sorted(filenames):
            if name in IGNORED_BASENAMES or name.startswith(IGNORED_PREFIXES):
                continue
            if name in SELF_DESCRIBING or name.endswith(".lock"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, ROOT)
            if rel in offi:
                # Re-hashing 1.07M files on every launch is not a gate anyone would keep, so
                # the default verifies presence for all and digests a fixed pseudo-random
                # sample; --deep digests everything. The sample seed is fixed so two runs
                # check the same files and a drifting file cannot hide behind fresh luck.
                if deep or rel in sampled:
                    (off_ok if _sha256(path) == offi[rel] else mismatch).append(rel)
                else:
                    off_ok.append(rel)
            elif rel in expl:
                (expl_ok if _sha256(path) == expl[rel] else mismatch).append(rel)
            elif name in known:
                (ok if _sha256(path) == known[name] else mismatch).append(rel)
            elif rel.startswith(EXPLORATORY_ROOT + os.sep):
                unknown.append(rel)      # inside data/exploratory but not manifested
            else:
                unknown.append(rel)
    return ok, unknown, mismatch, expl_ok, off_ok, missing_from_disk(known_paths(known), expl, offi)


def known_paths(known: Dict[str, str]) -> Dict[str, str]:
    """The seal is keyed by basename, so it cannot be checked in the manifest->disk
    direction without a path. Returns an empty map rather than inventing one."""
    return {}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA)
    p.add_argument("--deep", action="store_true",
                   help="digest every official-tree file instead of a fixed sample")
    p.add_argument("--allow-unmanifested", action="store_true",
                   help="proceed despite files data/ that the seal does not account for")
    a = p.parse_args(argv)

    if not os.path.isdir(a.data_dir):
        print(f"[data-hygiene] {a.data_dir} does not exist; nothing to check")
        return 0
    ok, unknown, mismatch, expl_ok, off_ok, gone = scan(a.data_dir, a.deep)
    if off_ok:
        mode = "every file" if a.deep else f"presence + {SAMPLE_N} sampled digests"
        print(f"[data-hygiene] {len(off_ok):,} official-tree file(s) accounted for by "
              f"CHECKSUMS.sha256 ({mode})")
    if gone:
        print(f"[data-hygiene] {len(gone)} MANIFESTED FILE(S) NOT ON DISK — a manifest is "
              f"describing files that are gone:")
        for r in gone:
            print(f"    {r}")
        print("[data-hygiene] this is a warning, not a refusal: a deletion may be "
              "deliberate. It is reported because a guard that only walks what exists "
              "cannot see one at all.")
    print(f"[data-hygiene] {len(ok)} manifested file(s) verified by sha256")
    if expl_ok:
        print(f"[data-hygiene] {len(expl_ok)} exploratory-class file(s) verified "
              f"[EXPLORATORY-UNVERIFIED-PROVENANCE] — recorded bytes, unknown origin; "
              f"never a registered input:")
        for r in expl_ok:
            print(f"    {r}")

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
