"""Assemble the hand-off bundle for the independent local reviewer.

Ships the raw evidence, the sealed mask manifest, the recomputation spec, and the
server's own analysis output to compare against. It deliberately does NOT ship
``src/analysis`` -- the reviewer's job is an independent reimplementation, and reading
ours first would defeat it. Everything shipped is checksummed into BUNDLE.sha256.

  python scripts/make_reviewer_bundle.py            # requires all 48 runs terminal
  python scripts/make_reviewer_bundle.py --allow-partial
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")
EXPECTED_RUNS = 48
EXPECTED_MASKS = 24


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def build(out_dir: str, allow_partial: bool = False) -> dict:
    terminal = sorted(d for d in os.listdir(RUNS)
                      if os.path.isfile(os.path.join(RUNS, d, "TERMINAL.json")))
    if len(terminal) != EXPECTED_RUNS and not allow_partial:
        raise SystemExit(f"{len(terminal)}/{EXPECTED_RUNS} runs terminal; "
                         f"re-run when the sweep finishes or pass --allow-partial")

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    shipped: List[str] = []

    for run_id in terminal:
        shipped.append(_copy(os.path.join(RUNS, run_id, "metrics.jsonl"),
                             os.path.join(out_dir, "metrics", run_id, "metrics.jsonl")))
        shipped.append(_copy(os.path.join(RUNS, run_id, "TERMINAL.json"),
                             os.path.join(out_dir, "metrics", run_id, "TERMINAL.json")))
    shipped.append(_copy(os.path.join(RUNS, "sweep_log.jsonl"),
                         os.path.join(out_dir, "metrics", "sweep_log.jsonl")))

    manifest_src = os.path.join(ROOT, "results", "noise_masks", "MANIFEST.json")
    manifest = json.load(open(manifest_src))
    if len(manifest) != EXPECTED_MASKS and not allow_partial:
        raise SystemExit(f"mask manifest holds {len(manifest)}/{EXPECTED_MASKS} entries; "
                         f"run scripts/verify_masks.py --update first")
    shipped.append(_copy(manifest_src, os.path.join(out_dir, "noise_masks", "MANIFEST.json")))

    for doc in ("analysis_protocol.md", "forward_pass_schema.md",
                "E_coverage_check.md", "G2_design_memo.md",
                "framing_prereg.md", "framing_preregistration.md",
                "g2_pin_requests.md", "classification_record.md"):
        src = os.path.join(ROOT, "docs", doc)
        if os.path.isfile(src):
            shipped.append(_copy(src, os.path.join(out_dir, "docs", doc)))
    shipped.append(_copy(os.path.join(ROOT, "configs", "base.yaml"),
                         os.path.join(out_dir, "base.yaml")))

    report_dir = os.path.join(ROOT, "results", "report")
    for name in ("analysis_values.json", "verdict.json", "G1_report.md",
                 "G1_independent_verification_report.md",
                 "b2_trajectory_identity.json", "b2_trajectory_identity_report.md",
                 "b2_negative_control.md", "full_d_exploratory.json",
                 "coarse_d_exploratory.json", "label_wave_baseline.json",
                 "sensitivity_exploratory.json",
                 "E_tgrid.json", "C1_noisyval.json", "B_representation.json",
                 "classification_verification.json",
                 "classification_verification_record.json"):
        src = os.path.join(report_dir, name)
        if os.path.isfile(src):
            shipped.append(_copy(src, os.path.join(out_dir, "server_analysis", name)))

    fwd_manifest = os.path.join(ROOT, "results", "forward_b2", "MANIFEST.sha256")
    if os.path.isfile(fwd_manifest):
        shipped.append(_copy(fwd_manifest,
                             os.path.join(out_dir, "forward_b2_MANIFEST.sha256")))

    checksums = {os.path.relpath(p, out_dir): _sha256(p) for p in sorted(shipped)}
    with open(os.path.join(out_dir, "BUNDLE.sha256"), "w") as fh:
        for rel, digest in checksums.items():
            fh.write(f"{digest}  {rel}\n")

    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(_readme(len(terminal), len(manifest), len(checksums)))

    return dict(out_dir=out_dir, runs=len(terminal), masks=len(manifest),
                files=len(checksums) + 2)


def _readme(n_runs: int, n_masks: int, n_files: int) -> str:
    return f"""# G1 reviewer bundle

Raw evidence from the server sweep, for an independent recomputation of the G1
analysis. Contents ({n_files} files, all checksummed in `BUNDLE.sha256`):

| path | what |
|---|---|
| `docs/analysis_protocol.md` | the recomputation spec — read this first |
| `docs/forward_pass_schema.md` | output schema of the B2 forward pass |
| `docs/E_coverage_check.md` | coverage check for baseline E |
| `docs/G2_design_memo.md` | G2 feasibility memo, including the E preregistration amendment |
| `server_analysis/b2_trajectory_identity*` | B2 rerun certified bit-identical to G1 (json + md) |
| `server_analysis/b2_negative_control.md` | proof the identity comparator detects deviations |
| `server_analysis/full_d_exploratory.json` | full-D (exploratory) |
| `forward_b2_MANIFEST.sha256` | sha256 of every B2 forward-pass output array |
| `docs/framing_prereg.md` | the registered classification rule (thresholds finalized) |
| `docs/framing_preregistration.md` | its pending-status record, open items and scope caveat |
| `docs/g2_pin_requests.md` | the C1 and (b) pins, with what was unpinned and why |
| `docs/classification_record.md` | the adjudication: S3 -> audit ruling, E adjudication, statistic pin |
| `server_analysis/E_tgrid.json` | selector E, T-grid |
| `server_analysis/C1_noisyval.json` | selector C1, noisy-validation |
| `server_analysis/B_representation.json` | selector (b), effective rank |
| `server_analysis/classification_verification*.json` | independent recomputation of every adjudicated count |
| `metrics/<run_id>/metrics.jsonl` | {n_runs} raw per-epoch logs (1 metadata line + 120 epoch lines) |
| `metrics/<run_id>/TERMINAL.json` | completion marker per run |
| `metrics/sweep_log.jsonl` | run -> GPU assignment and start/end timestamps |
| `noise_masks/MANIFEST.json` | sealed sha256 of all {n_masks} noisy-label masks |
| `base.yaml` | the config the sweep ran under |
| `server_analysis/analysis_values.json` | the server's numbers at full float precision — the comparison target |
| `server_analysis/verdict.json` | the server's sealed-rule verdict |
| `server_analysis/G1_report.md` | the server's rendered report |
| `server_analysis/G1_independent_verification_report.md` | an independent reimplementation's verdict on the above |

`G1_independent_verification_report.md` was **delivered by paste from the verifying
session; integrity confirmed by sha256**
(`2eb6013a2c116abe4f756c64d7bed1a14ef3c33a39cc89db69151926e0001f69`, 3053 bytes) and
stored byte-exact without edits. It reports bit-exact agreement with the server's
analysis on every checked quantity — oracle epochs, all floats, all BCa bounds, and
every verdict counter — reached from `analysis_protocol.md` alone.

`src/analysis` is deliberately **not** included: the point of the exercise is an
independent reimplementation, and the spec in `analysis_protocol.md` is meant to be
sufficient on its own. If it is not, that gap is itself a finding worth reporting.

Verify integrity before starting:

    sha256sum -c BUNDLE.sha256

Tolerances and the shared bootstrap-seed protocol are in `analysis_protocol.md`
sections 6 and 8. Point estimates should agree to 1e-9 and oracle epochs exactly;
BCa bounds should agree exactly once the seed protocol is matched.
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.join(ROOT, "results", "reviewer_bundle"))
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args(argv)
    info = build(a.out_dir, a.allow_partial)
    print(f"[bundle] {info['runs']} runs, {info['masks']} masks, {info['files']} files "
          f"-> {info['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
