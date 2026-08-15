# No Clean Oracle — audit code

Research audit code for the **No Clean Oracle** project: a measurement-only study of
whether checkpoints optimal for in-distribution accuracy, worst-class accuracy, and
out-of-distribution reliability systematically diverge under noisy-label training, when
no clean validation set is available to choose between them.

This repository is the code and specification record. It is not a dataset release and
not a results release.

## Datasets are not redistributed

No dataset payload is tracked here, and none ever has been. CIFAR-10/100, the CIFAR-N
human-annotation files, SVHN and any corruption sets must be obtained from their official
sources; `results/cifar_n_masks/MANIFEST.json` records the source URL and sha256 of each
file the audit consumed, so a third party can verify they obtained the same bytes without
this repository shipping them.

Clothing1M, referenced in the Tier 2 amendment, enters the project only through its
official agreement. No mirror, scrape, or derived copy is used or accepted.

## Results are fingerprint-attributed, not tracked

Training logs, checkpoints, logits, embeddings and analysis outputs live under
`results/`, which is git-ignored. They are attributed by fingerprint instead:

- every run records a dual-axis `code_stamp` — the git HEAD it launched from plus the
  sha256 of every source module the process actually imported (`src/provenance.py`);
- sealed data inputs carry a manifest of sha256 digests, re-verified on every load;
- produced artifact trees carry a sorted per-file `MANIFEST.sha256`, so one hash attests
  a whole tree.

A claim in the paper therefore cites a commit and a digest rather than a file that has to
be trusted to be the one that ran.

## Layout

| path | contents |
|---|---|
| `src/` | trainer, learners, evaluation, noise handling, analysis library |
| `scripts/` | run/sweep/forward-pass entry points, launchers, verifiers |
| `configs/` | sealed experiment configurations |
| `docs/` | preregistrations, protocol, amendments, adjudication records |
| `tests/` | unit tests |

## Preregistration

The audit is preregistered. `docs/` holds the registered documents and the protocol
rules that govern them, including the rule that corrections are follow-up commits and
never history rewrites, so a document's history is auditable rather than tidy.

## License

MIT — see `LICENSE`.
