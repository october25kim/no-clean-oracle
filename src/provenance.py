"""Executing-code fingerprint, captured at process start.

An artifact that records only a commit hash is trusting that the working tree matched
that commit when the process launched -- which is exactly the question that had to be
reconstructed by hand for the first forward pass (refactor mtime vs container start vs
commit time). Stamping the fingerprint into the artifact removes the reconstruction:
the file says which code produced it.

Both halves are recorded because either alone can mislead. The commit locates the code
in history; the per-module sha256 set proves the tree actually matched it, and still
identifies the code if it did not.

Call ``code_stamp()`` once at the start of ``main()``, after imports, so the module set
reflects what the process really loaded.

Injected git state (``git_source: injected``) reflects the host tree at launcher
invocation, not at process start; the gap is closed operationally by R2 (pre-launch
commit, clean tree) rather than by the stamp itself.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", ROOT, *args], text=True,
                                       stderr=subprocess.DEVNULL, timeout=30).strip()
    except Exception:
        return None


def _sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def code_stamp() -> Dict:
    """Fingerprint of the code this process is executing, right now.

    ``git_available`` distinguishes "the tree was clean" from "git could not be asked".
    The training image has no git binary, so without this flag a container run would
    report ``git_tree_dirty: false`` for the second reason while reading like the first,
    and an R2 guard testing only that field would pass vacuously. Guards must test
    ``git_available and not git_tree_dirty``; when git is unavailable the
    ``modules_sha256`` set is the operative attribution and is always present.
    """
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    git_available = head is not None and status is not None
    git_source = "local git" if git_available else None
    if not git_available and os.environ.get("G1_GIT_HEAD"):
        # The training image has no git binary, so the launcher captures the host's
        # state at launch and passes it in. Marked as injected, never conflated with a
        # local query.
        head = os.environ["G1_GIT_HEAD"]
        status = os.environ.get("G1_GIT_DIRTY_PATHS", "")
        git_available = True
        git_source = "injected by launcher (G1_GIT_HEAD)"
    dirty = status or ""
    modules: Dict[str, str] = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None) or ""
        if f.endswith(".py") and os.path.abspath(f).startswith(ROOT + os.sep):
            digest = _sha256(f)
            if digest:
                modules[os.path.relpath(os.path.abspath(f), ROOT)] = digest
    entry = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    if entry.startswith(ROOT + os.sep) and os.path.isfile(entry):
        modules[os.path.relpath(entry, ROOT)] = _sha256(entry)

    combined = hashlib.sha256(
        "".join(f"{k}:{v}\n" for k, v in sorted(modules.items())).encode()).hexdigest()
    return dict(
        git_available=git_available,
        git_source=git_source,
        git_head=head,
        git_head_short=(head[:7] if head else None),
        git_tree_dirty=(bool(dirty) if git_available else None),
        git_dirty_paths=sorted(line[3:] for line in dirty.splitlines()) if dirty else [],
        modules_sha256=modules,
        modules_combined_sha256=combined,
        captured_at_unix=round(time.time(), 3),
        captured_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
