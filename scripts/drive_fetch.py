"""Fetch an authorized Google Drive folder that requires a resource key.

Written because ``gdown`` cannot fetch this link **in either mode**, and the reason is
narrow and specific rather than a permissions problem:

* ``gdown.download_folder`` derives the folder id with
  ``urlparse(url).path.rstrip("/").split("/")[-1]``, which keeps only the path and discards
  the query string. The ``resourcekey`` never reaches the request it builds, so
  ``embeddedfolderview?id=<id>`` returns **401**.
* ``gdown.download`` drops it the same way -- its own failure message echoes the URL back
  with ``resourcekey`` already missing.
* ``grep -rn resourcekey`` over the installed gdown finds **zero** matches, and 6.1.0 is the
  newest release, so there is no version to upgrade to.

Google requires the resource key for anonymous access to objects created before its
September 2021 link-sharing change, which is every object in a 2015 folder. Supplying it is
not a permission bypass: the key is issued by Google precisely so that a link holder can
read the object, and it arrives as part of the link the owner provided. Adding it takes the
same request from 401 to 200 with nothing else changed -- verified before this module was
written.

This is therefore **not** a mirror, a re-upload, or a scrape. It is the official folder,
fetched from ``drive.google.com`` and ``drive.usercontent.google.com``, using the access
parameter the authorized link carries. The one thing it does that gdown would have done for
us is the large-file confirm-token exchange, which Drive requires for anything past its
virus-scan threshold.

**Identifiers are secrets here.** A file id together with its resource key reconstructs the
link, so neither is ever written to disk by this module -- not to a log, not to a manifest.
They live in memory for the length of the fetch. Callers must keep them out of anything that
anchors.
"""
from __future__ import annotations

import os
import re
import sys
import time
import urllib.parse
from typing import Dict, List, Optional, Tuple

import requests

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
VIEW = "https://drive.google.com/embeddedfolderview"
UC = "https://drive.google.com/uc"
FOLDER_MIME_HINT = "drive/folders/"

# Defence in depth. Nothing in this module builds a printable URL -- every emission uses a
# relative path and an HTTP status -- but that is a property of the current code, not a
# guarantee, and the gdown path in the caller is redacted while this one was not. A future
# edit that puts a URL into an error message would leak it silently. Route every emission
# through a redactor so the property is enforced rather than maintained by attention.
_URLISH = re.compile(r"(https?://\S+|drive\.google\.com/\S*|[?&](?:id|resourcekey)=[\w-]+|/d/[\w-]{20,})")


def redact(text: str) -> str:
    return _URLISH.sub("[REDACTED-URL]", text or "")


def _say(msg: str) -> None:
    print(redact(msg))


_ENTRY = re.compile(
    r'href="https://drive\.google\.com/(file/d/|drive/folders/)([\w-]+)[^"]*?resourcekey=([\w-]+)')
_TITLE = re.compile(r'<div class="flip-entry-title">([^<]+)</div>')
_FORM = re.compile(r'<form[^>]+action="([^"]+)"')
_HIDDEN = re.compile(r'<input type="hidden" name="([^"]+)" value="([^"]*)"')


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def list_folder(sess: requests.Session, folder_id: str, key: str) -> List[dict]:
    """One folder level: [{title, id, key, is_folder}]. Raises on a non-200."""
    r = sess.get(f"{VIEW}?{urllib.parse.urlencode({'id': folder_id, 'resourcekey': key})}",
                 timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"folder listing failed with HTTP {r.status_code} "
                           f"(a missing or wrong resource key returns 401)")
    kinds = _ENTRY.findall(r.text)
    titles = [t.strip() for t in _TITLE.findall(r.text)]
    if len(kinds) != len(titles):
        raise RuntimeError(f"listing parse mismatch: {len(titles)} titles vs {len(kinds)} links")
    return [dict(title=t, id=i, key=k, is_folder=(kind == FOLDER_MIME_HINT))
            for t, (kind, i, k) in zip(titles, kinds)]


def walk(sess: requests.Session, folder_id: str, key: str,
         prefix: str = "") -> List[Tuple[str, str, str]]:
    """Depth-first (relative_path, id, key) for every FILE beneath the folder."""
    out: List[Tuple[str, str, str]] = []
    for e in list_folder(sess, folder_id, key):
        rel = os.path.join(prefix, e["title"])
        if e["is_folder"]:
            out.extend(walk(sess, e["id"], e["key"], rel))
        else:
            out.append((rel, e["id"], e["key"]))
    return out


def _resolve(sess: requests.Session, fid: str, key: str):
    """GET the file; if Drive answers with its confirm page, submit that form."""
    url = f"{UC}?{urllib.parse.urlencode({'id': fid, 'export': 'download', 'resourcekey': key})}"
    r = sess.get(url, stream=True, allow_redirects=True, timeout=120)
    if "text/html" not in r.headers.get("content-type", ""):
        return r
    body = r.text
    r.close()
    action = _FORM.search(body)
    fields = dict(_HIDDEN.findall(body))
    if not action or not fields:
        raise RuntimeError("Drive returned HTML with no confirm form; the file may be "
                           "unavailable or the resource key wrong")
    return sess.get(action.group(1), params=fields, stream=True,
                    allow_redirects=True, timeout=120)


def fetch_file(sess: requests.Session, rel: str, fid: str, key: str, dest_root: str,
               quiet: bool = False) -> str:
    """Stream one file to dest_root/rel via a temp file and an atomic rename.

    The temp name carries the pid, and the rename only happens once the body is fully
    written, so an interrupted fetch leaves a partial temp rather than a short file that
    looks complete -- the same failure shape as the 0-byte TERMINAL.
    """
    dest = os.path.join(dest_root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = f"{dest}.part.{os.getpid()}"
    r = _resolve(sess, fid, key)
    if r.status_code != 200:
        raise RuntimeError(redact(f"{rel}: HTTP {r.status_code}"))
    total = int(r.headers.get("content-length") or 0)
    got, t0, last = 0, time.time(), 0.0
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(1 << 22):
            if not chunk:
                continue
            fh.write(chunk)
            got += len(chunk)
            now = time.time()
            if not quiet and now - last > 30:
                last = now
                pct = f"{100 * got / total:5.1f}%" if total else "  ?  "
                _say(f"    {rel}: {pct} {got / 2**30:7.2f} GiB "
                      f"{got / 2**20 / max(now - t0, 1e-9):6.1f} MiB/s")
    r.close()
    if total and got != total:
        os.unlink(tmp)
        raise RuntimeError(f"{rel}: short read, {got} of {total} bytes")
    os.replace(tmp, dest)
    if not quiet:
        _say(f"    {rel}: done, {got / 2**30:.2f} GiB in {time.time() - t0:.0f}s")
    return dest


def fetch_folder(folder_id: str, key: str, dest_root: str, quiet: bool = False) -> List[str]:
    sess = _session()
    items = walk(sess, folder_id, key)
    _say(f"[drive] {len(items)} file(s) to fetch")
    for rel, _i, _k in items:
        _say(f"    - {rel}")
    written = []
    for rel, fid, k in items:
        dest = os.path.join(dest_root, rel)
        if os.path.exists(dest):
            _say(f"    {rel}: already present, skipping")
            written.append(dest)
            continue
        written.append(fetch_file(sess, rel, fid, k, dest_root, quiet=quiet))
    return written
