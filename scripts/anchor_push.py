"""Anchor push: publish the internal tracked tree as a snapshot, with parity enforced.

Replaces the ad-hoc shell that produced anchors 1-15. That shell built the snapshot with
``git archive HEAD | tar -x`` and then ``git add -A``, which silently dropped five tracked
files: the fresh snapshot repository re-applies ``.gitignore`` to paths that are untracked
*there*, and the project's rules ``results/`` and ``data/`` match
``results/cifar_n_masks/MANIFEST.json`` and ``src/data/*`` -- the latter because a bare
``data/`` pattern matches a directory of that name at any depth, not just at the root.
Internally those files are tracked and therefore exempt; in a new repository they are not.

The result was a public anchor missing four source modules and the CIFAR-N manifest, which
is the file the README points at for dataset provenance. Nothing was wrong with the files
that did cross -- every shared path was byte-identical -- but the anchor did not reproduce
the code that ran, which is the one thing an anchor exists to do.

Two guards now stand between a snapshot and a push:

* ``git add -A -f`` so an ignore rule cannot drop a file the internal repository tracks;
* an explicit **file-set parity check** against ``git ls-files`` in the internal repository,
  which refuses the push on any difference in either direction. The parity check is the
  real guard -- the force-add fixes the known cause, the check catches the next one.

R6-guard still applies: the remote HEAD must equal the recorded baseline before anything
is pushed.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE = "https://github.com/october25kim/no-clean-oracle"
CHAIN = os.path.join(ROOT, "docs", "ANCHOR_CHAIN.md")
IDENTITY = ("october25kim", "october25kim@users.noreply.github.com")


def run(cmd, cwd=None, check=True) -> str:
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and p.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}\n{p.stdout}{p.stderr}")
    return p.stdout.strip()


def recorded_baseline() -> str:
    """The R6-guard baseline, read from the fenced block in ANCHOR_CHAIN.md."""
    txt = open(CHAIN).read()
    m = re.search(r"R6-guard baseline.*?```\s*([0-9a-f]{40})\s*```", txt, re.S)
    if not m:
        raise SystemExit("no R6-guard baseline found in docs/ANCHOR_CHAIN.md")
    return m.group(1)


def record_push(public: str, internal: str, subject: str) -> None:
    """Append the push row and advance the baseline, in the file, immediately.

    This used to be a printed reminder to do it by hand, and the hand step was missed
    after the T-EMIT-3 registration push: the baseline sat one push behind and the guard
    refused the next attempt. The guard doing that is the fix working — D-8b made staleness
    loud instead of silent — but the reminder was the wrong mechanism, because a step that
    only a human remembers is a step that eventually is not taken. The row lands in the
    working tree and rides the following snapshot, which is the same one-push lag the
    manual version had, minus the forgetting.
    """
    txt = open(CHAIN).read()
    rows = re.findall(r"^\| (\d+) \| `[0-9a-f]{40}`", txt, re.M)
    n = max(int(x) for x in rows) + 1 if rows else 1
    stamp = subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                           text=True, capture_output=True).stdout.strip()
    subj = subject.replace("Anchor snapshot: ", "").strip()
    last = sorted(((int(m.group(1)), m.group(0)) for m in
                   re.finditer(r"^\| \d+ \| `[0-9a-f]{40}` \|.*$", txt, re.M)),
                  key=lambda t: t[0])[-1][1]
    txt = txt.replace(last, f"{last}\n| {n} | `{public}` | {stamp} | `{internal[:7]}` | {subj} |")
    txt = re.sub(r"(R6-guard baseline.*?```\s*)[0-9a-f]{40}(\s*```)",
                 rf"\g<1>{public}\g<2>", txt, flags=re.S)
    open(CHAIN, "w").write(txt)
    print(f"[anchor] ANCHOR_CHAIN row {n} recorded; baseline advanced to {public}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--message", required=True, help="snapshot commit subject")
    p.add_argument("--work", default=os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "anchor_push_work"))
    p.add_argument("--dry-run", action="store_true",
                   help="build and check parity, then stop without pushing")
    a = p.parse_args(argv)

    dirty = run(["git", "-C", ROOT, "status", "--porcelain"])
    if dirty:
        raise SystemExit(f"R2: internal tree is dirty; commit first\n{dirty}")
    internal = run(["git", "-C", ROOT, "rev-parse", "HEAD"])
    want = sorted(run(["git", "-C", ROOT, "ls-files"]).splitlines())

    base = recorded_baseline()
    remote = run(["git", "ls-remote", REMOTE, "refs/heads/master"]).split()[0]
    print(f"[anchor] R6-guard baseline {base}\n[anchor] remote HEAD      {remote}")
    if remote != base:
        raise SystemExit("R6-guard: remote HEAD is not our recorded baseline — a foreign "
                         "writer reached the anchor. STOP and report; do not push.")

    if os.path.exists(a.work):
        subprocess.run(["rm", "-rf", a.work], check=True)
    run(["git", "clone", "-q", REMOTE, a.work])
    for entry in os.listdir(a.work):
        if entry != ".git":
            subprocess.run(["rm", "-rf", os.path.join(a.work, entry)], check=True)
    archive = subprocess.run(["git", "-C", ROOT, "archive", "HEAD"],
                             check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", a.work], input=archive, check=True)

    run(["git", "-C", a.work, "config", "user.name", IDENTITY[0]])
    run(["git", "-C", a.work, "config", "user.email", IDENTITY[1]])
    run(["git", "-C", a.work, "add", "-A", "-f"])     # -f: ignore rules must not drop
    got = sorted(run(["git", "-C", a.work, "ls-files"]).splitlines())

    missing = [f for f in want if f not in set(got)]
    extra = [f for f in set(got) if f not in set(want)]
    print(f"[anchor] internal tracks {len(want)} files; snapshot stages {len(got)}")
    if missing or extra:
        for f in missing:
            print(f"  MISSING from snapshot: {f}")
        for f in extra:
            print(f"  EXTRA in snapshot:     {f}")
        raise SystemExit("file-set parity failed — refusing to push a snapshot that is "
                         "not the internal tracked tree")
    print("[anchor] file-set parity OK")

    if a.dry_run:
        print("[anchor] --dry-run: not pushing")
        return 0

    if not run(["git", "-C", a.work, "status", "--porcelain"]):
        print("[anchor] snapshot is identical to the current public HEAD; nothing to push")
        return 0
    run(["git", "-C", a.work, "commit", "-q", "-m",
         f"{a.message} (internal {internal[:7]} lineage)"])
    run(["git", "-C", a.work, "push", "-q", "origin", "master"])
    new = run(["git", "-C", a.work, "rev-parse", "HEAD"])
    print(f"[anchor] pushed. new public HEAD {new}")
    record_push(new, internal, a.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
