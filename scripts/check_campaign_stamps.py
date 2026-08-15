"""Campaign-wide code attribution: did every run in a results dir execute the same code?

The obvious check -- "is modules_combined_sha256 the same everywhere?" -- is wrong, and
the Tier 1 launcher header asserted it before this script existed. ``code_stamp()``
digests the modules the process actually imported, and ``build_loss`` imports the learner
lazily inside its branch, so a CE run stamps 16 modules and an ELR run stamps 17. The
combined digest is constant within a learner and differs between learners for a reason
that has nothing to do with code drift.

The invariant that does hold, and that this checks, is per-module: any module that
appears in more than one run must carry the same sha256 in all of them. That catches the
thing the combined digest was meant to catch -- a source file edited mid-campaign, so
that later runs trained under different code -- without firing on the learner axis.

Read-only. Exits 1 on any drift.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_stamps(results_dir: str) -> Dict[str, Tuple[str, dict]]:
    """{run_id: (learner, code_stamp)} from each run's metadata line."""
    out: Dict[str, Tuple[str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "metrics.jsonl"))):
        run_id = os.path.basename(os.path.dirname(path))
        with open(path) as fh:
            first = fh.readline().strip()
        if not first:
            continue
        try:
            meta = json.loads(first)
        except json.JSONDecodeError:
            continue
        if meta.get("record") != "metadata" or "code_stamp" not in meta:
            continue
        out[run_id] = (meta.get("learner", "?"), meta["code_stamp"])
    return out


def check(stamps: Dict[str, Tuple[str, dict]]) -> List[str]:
    """Return a list of drift descriptions; empty means the campaign is homogeneous."""
    by_module: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for run_id, (_learner, cs) in stamps.items():
        for mod, digest in cs.get("modules_sha256", {}).items():
            by_module[mod][digest].append(run_id)

    problems: List[str] = []
    for mod in sorted(by_module):
        variants = by_module[mod]
        if len(variants) > 1:
            detail = "; ".join(
                f"{d[:12]}… in {len(runs)} run(s) e.g. {sorted(runs)[0]}"
                for d, runs in sorted(variants.items()))
            problems.append(f"{mod}: {len(variants)} distinct sha256 -- {detail}")

    # git_tree_dirty must be a real answer, never the fail-open "git could not be asked"
    for run_id, (_l, cs) in sorted(stamps.items()):
        if not cs.get("git_available"):
            problems.append(f"{run_id}: git_available false -- stamp cannot attest the tree")
        elif cs.get("git_tree_dirty"):
            problems.append(f"{run_id}: launched from a dirty tree "
                            f"({', '.join(cs.get('git_dirty_paths', []))})")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=os.path.join(ROOT, "results", "runs_ext"))
    a = p.parse_args(argv)

    stamps = load_stamps(a.results_dir)
    if not stamps:
        print(f"[stamps] no stamped runs under {a.results_dir}")
        return 0

    by_combined: Dict[str, List[str]] = defaultdict(list)
    by_learner_combined: Dict[str, set] = defaultdict(set)
    for run_id, (learner, cs) in sorted(stamps.items()):
        by_combined[cs["modules_combined_sha256"]].append(run_id)
        by_learner_combined[learner].add(cs["modules_combined_sha256"])

    print(f"[stamps] {len(stamps)} run(s) under {os.path.relpath(a.results_dir, ROOT)}")
    print("[stamps] R4 class: trajectory-identity (exact sha256 equality; tolerance 0). "
          "R5 invariant: every module in >=2 runs byte-identical everywhere.")
    for learner in sorted(by_learner_combined):
        digests = sorted(by_learner_combined[learner])
        runs = sum(1 for _r, (l, _c) in stamps.items() if l == learner)
        mark = "ok" if len(digests) == 1 else f"DRIFT ({len(digests)} digests)"
        print(f"    {learner:4s} {runs:3d} run(s)  combined {digests[0][:16]}…  {mark}")

    heads = sorted({cs["git_head_short"] for _r, (_l, cs) in stamps.items()})
    print(f"[stamps] git_head(s): {', '.join(h or 'none' for h in heads)}")

    problems = check(stamps)
    if problems:
        print(f"[stamps] {len(problems)} PROBLEM(S):")
        for prob in problems:
            print(f"    {prob}")
        return 1
    n_shared = len({m for _r, (_l, cs) in stamps.items() for m in cs["modules_sha256"]})
    print(f"[stamps] every one of {n_shared} module(s) is byte-identical wherever it "
          f"appears; no code drift across the campaign")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
