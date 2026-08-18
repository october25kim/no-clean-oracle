"""Ingest the officially obtained Clothing1M release.  [VERIFIED-OFFICIAL on PASS]

This implements steps 1-4 of the ingestion trigger. It does **not** classify anything
`[VERIFIED-OFFICIAL]` by itself: it verifies structure and stops on any mismatch, and the
classification plus the Tier-2 registration draft are the operator's step 5.

**The link is never written down.** It is read from stdin or from an environment variable,
never taken as a command-line argument, because argv is visible in ``ps`` output and lands
in shell history. ``gdown`` echoes the URL in its own progress output, so both of its
streams are captured and passed through a redactor before anything is printed or logged; the
raw streams are never surfaced. ``PROVENANCE.md`` records the channel, the date and the
sentence "link held privately by the owner", and nothing that could reconstruct the URL.

What this file's checksums are and are not: they are **our** canonical hashes, computed here
and used from here on to detect drift. No official checksum for Clothing1M exists, so they
cannot validate the download against the canonical release. Structure verification counts
records; it is a structural check, and a re-upload with the right counts would pass it. The
`[VERIFIED-OFFICIAL]` classification therefore rests on the **channel** -- author-direct to
the owner -- and not on anything this script can prove. That distinction is written into
PROVENANCE.md rather than left for a reader to infer.

Published counts checked in step 3, from the trigger:

* 14 classes
* noisy train ~1,000,000 (tolerance applies; the release is documented as approximate)
* clean train / val / test exactly 47,570 / 14,313 / 10,526
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import time
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "data", "clothing1m_official")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "raw")
EXTRACTED = os.path.join(BASE, "extracted")
CHECKSUMS = os.path.join(BASE, "CHECKSUMS.sha256")
PROVENANCE = os.path.join(BASE, "PROVENANCE.md")

EXPECT_CLASSES = 14
EXPECT_NOISY_TRAIN = 1_000_000
NOISY_TOLERANCE = 0.01                      # the release documents this count as approximate
EXPECT_EXACT = {"clean_train": 47_570, "clean_val": 14_313, "clean_test": 10_526}

KEY_LISTS = {
    "noisy_train": ("noisy_train_key_list.txt",),
    "clean_train": ("clean_train_key_list.txt",),
    "clean_val": ("clean_val_key_list.txt",),
    "clean_test": ("clean_test_key_list.txt",),
}
CATEGORY_FILES = ("category_names_eng.txt", "category_names_chn.txt")

_URLISH = re.compile(r"(https?://\S+|drive\.google\.com/\S*|[?&]id=[\w-]+|/d/[\w-]{20,})")


def redact(text: str) -> str:
    """Remove anything that could reconstruct the link from a captured stream."""
    return _URLISH.sub("[REDACTED-URL]", text or "")


def read_url(args) -> str:
    """stdin or env only. Never argv: argv is in ``ps`` and in shell history."""
    if args.url_env:
        url = os.environ.get(args.url_env, "")
        if not url:
            raise SystemExit(f"environment variable {args.url_env} is empty")
        return url.strip()
    if sys.stdin.isatty():
        raise SystemExit("no link on stdin; pipe it in or use --url-env")
    url = sys.stdin.read().strip()
    if not url:
        raise SystemExit("empty link on stdin")
    return url


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def free_bytes(path: str) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def download(url: str, folder_mode: bool) -> None:
    """gdown into RAW. Both streams are captured and redacted before anyone sees them."""
    os.makedirs(RAW, exist_ok=True)
    cmd = ["python", "-m", "gdown"]
    cmd += (["--folder", url, "-O", RAW] if folder_mode else [url, "-O", RAW + os.sep])
    print(f"[ingest] downloading into {os.path.relpath(RAW, ROOT)} "
          f"({'folder' if folder_mode else 'file'} mode)", flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True)
    out = redact(p.stdout)[-4000:]
    err = redact(p.stderr)[-4000:]
    if p.returncode != 0:
        raise SystemExit(f"gdown failed (rc={p.returncode})\n{out}\n{err}")
    print(redact(err.strip()[-1500:]) or "[ingest] gdown finished")


def lock_or_exit(tag: str) -> Optional[int]:
    """Single-elector lock: pid-suffixed temp + O_EXCL, the pattern that fixed the race."""
    lock = os.path.join(BASE, f".{tag}.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        print(f"[ingest] another elector holds {os.path.basename(lock)}; this process stands "
              f"down rather than racing it")
        return None
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def write_checksums(files: List[str]) -> Dict[str, str]:
    rows = {}
    with open(CHECKSUMS, "w") as fh:
        fh.write("# Our canonical hashes for the officially obtained Clothing1M release.\n")
        fh.write("# No official checksum exists, so these pin bytes going forward; they do\n")
        fh.write("# NOT validate this download against the canonical release.\n")
        for f in sorted(files):
            d = sha256(f)
            rel = os.path.relpath(f, BASE)
            rows[rel] = d
            fh.write(f"{d}  {rel}\n")
    print(f"[ingest] wrote {os.path.relpath(CHECKSUMS, ROOT)} ({len(rows)} file(s))")
    return rows


def find_one(root: str, names: Tuple[str, ...]) -> Optional[str]:
    for dirpath, _d, filenames in os.walk(root):
        for n in names:
            if n in filenames:
                return os.path.join(dirpath, n)
    return None


def count_lines(path: str) -> int:
    with open(path, "rb") as fh:
        return sum(1 for line in fh if line.strip())


def verify_structure(root: str) -> Tuple[bool, dict]:
    """Count records against the published counts. STOP on any mismatch."""
    report: Dict[str, object] = {}
    ok = True

    cat = find_one(root, CATEGORY_FILES)
    if cat is None:
        report["classes"] = "category names file not found"
        ok = False
    else:
        n = count_lines(cat)
        report["classes"] = dict(found=n, expected=EXPECT_CLASSES, file=os.path.relpath(cat, root))
        ok &= (n == EXPECT_CLASSES)

    for key, names in KEY_LISTS.items():
        f = find_one(root, names)
        if f is None:
            report[key] = f"{names[0]} not found"
            ok = False
            continue
        n = count_lines(f)
        if key == "noisy_train":
            lo = int(EXPECT_NOISY_TRAIN * (1 - NOISY_TOLERANCE))
            hi = int(EXPECT_NOISY_TRAIN * (1 + NOISY_TOLERANCE))
            good = lo <= n <= hi
            report[key] = dict(found=n, expected_approx=EXPECT_NOISY_TRAIN,
                               accepted_range=[lo, hi], ok=good)
        else:
            good = (n == EXPECT_EXACT[key])
            report[key] = dict(found=n, expected_exact=EXPECT_EXACT[key], ok=good)
        ok &= good

    return ok, report


def freeze(path: str, recursive: bool) -> None:
    if recursive:
        subprocess.run(["chmod", "-R", "a-w", path], check=True)
    else:
        for f in os.listdir(path):
            p = os.path.join(path, f)
            if os.path.isfile(p):
                os.chmod(p, 0o444)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url-env", help="env var holding the link (never argv)")
    p.add_argument("--folder", action="store_true", help="link is a Drive folder")
    p.add_argument("--drive-folder", action="store_true",
                   help="Drive folder needing a resource key; bypasses gdown, which drops it")
    p.add_argument("--skip-download", action="store_true",
                   help="raw/ is already populated; resume at checksums")
    p.add_argument("--extract", action="store_true",
                   help="run the single-elector extraction step")
    a = p.parse_args(argv)

    os.makedirs(BASE, exist_ok=True)

    if not a.skip_download:
        url = read_url(a)                       # never printed, never stored
        if a.drive_folder:
            # gdown cannot fetch this link in either mode: it derives the id from the URL
            # path and discards the query string, so `resourcekey` never reaches the
            # request and Drive answers 401. See scripts/drive_fetch.py. The id and key are
            # parsed here, held in memory, and never written anywhere -- together they
            # reconstruct the link.
            import drive_fetch
            q = urllib.parse.urlparse(url)
            fid = q.path.rstrip("/").split("/")[-1]
            key = urllib.parse.parse_qs(q.query).get("resourcekey", [""])[0]
            if not key:
                raise SystemExit("--drive-folder needs a link carrying ?resourcekey=")
            os.makedirs(RAW, exist_ok=True)
            drive_fetch.fetch_folder(fid, key, RAW)
            del fid, key
        else:
            download(url, a.folder)
        del url

    # Layer (b) covers archives and extracted content only (ruling 40-L1). macOS metadata is
    # never data, and a first pass swept four such files into CHECKSUMS.sha256, pinning
    # AppleDouble stubs as if they were part of the release. They are excluded here rather
    # than filtered later, so the manifest cannot contain them in the first place.
    raw_files = [os.path.join(dp, f) for dp, _d, fs in os.walk(RAW) for f in fs
                 if not (f.startswith("._") or f == ".DS_Store")]
    if not raw_files:
        raise SystemExit(f"{os.path.relpath(RAW, ROOT)} is empty; nothing to ingest")
    total = sum(os.path.getsize(f) for f in raw_files)
    print(f"[ingest] raw: {len(raw_files)} file(s), {total / 2**30:.2f} GiB")

    freeze(RAW, recursive=False)                # chmod 444 raw/*
    write_checksums(raw_files)

    if a.extract:
        avail = free_bytes(BASE)
        print(f"[ingest] free {avail / 2**30:.1f} GiB, archive {total / 2**30:.2f} GiB, "
              f"2x guard {2 * total / 2**30:.2f} GiB")
        if avail < 2 * total:
            raise SystemExit("free space is below 2x the archive size; aborting before "
                             "extraction rather than filling the filesystem")
        fd = lock_or_exit("extract")
        if fd is None:
            return 0
        os.makedirs(EXTRACTED, exist_ok=True)
        before = free_bytes(BASE)
        t0 = time.time()
        for f in sorted(raw_files):
            if f.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
                print(f"[ingest]   unpacking {os.path.basename(f)}", flush=True)
                shutil.unpack_archive(f, EXTRACTED)
        after = free_bytes(BASE)
        print(f"[ingest] extraction {time.time() - t0:.0f}s; disk used "
              f"{(before - after) / 2**30:.2f} GiB "
              f"(free {before / 2**30:.1f} -> {after / 2**30:.1f} GiB)")

        ok, report = verify_structure(EXTRACTED)
        print(json.dumps(report, indent=1))

        # Layer (b) is the canonical pin and covers archives AND extracted content
        # (ruling 40-L1). Written before the tree is frozen, over both roots, with macOS
        # metadata excluded at the source rather than filtered afterwards.
        ext_files = [os.path.join(dp, f) for dp, _d, fs in os.walk(EXTRACTED) for f in fs
                     if not (f.startswith("._") or f == ".DS_Store")]
        print(f"[ingest] layer (b): hashing {len(raw_files) + len(ext_files):,} files "
              f"({len(raw_files)} archives + {len(ext_files):,} extracted)", flush=True)
        write_checksums(raw_files + ext_files)

        freeze(EXTRACTED, recursive=True)       # chmod -R a-w
        os.close(fd)
        if not ok:
            raise SystemExit("STRUCTURE MISMATCH — stopping. The release does not match the "
                             "published counts; do not classify this VERIFIED-OFFICIAL.")
        print("[ingest] structure PASS against the published counts")

    print("[ingest] steps 1-4 complete. Classification and the Tier-2 registration draft "
          "are step 5 and are not done here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
