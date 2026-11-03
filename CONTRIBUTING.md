# Contributing

This is primarily a solo research repo, but issues and pull requests are
welcome, especially anything that catches a reproducibility bug, a doc/code
mismatch, or an unbacked claim.

## Before opening a PR

Every project in this repo keeps its own verification suite. For
[`FedGRAA/`](FedGRAA/), run all of these from `FedGRAA/` and make sure they exit 0:

```powershell
.venv313\Scripts\python.exe -m pytest tests --tb=short
.venv313\Scripts\python.exe scripts\check_stage_a_complete.py
.venv313\Scripts\python.exe scripts\check_stage_b_complete.py
.venv313\Scripts\python.exe scripts\check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6 TC --t4-m 3
.venv313\Scripts\python.exe scripts\check_stage_c_complete.py --tier T7 --seeds 0 1 2 3 4
.venv313\Scripts\python.exe scripts\audit_stage_a.py
.venv313\Scripts\python.exe scripts\audit_stage_b.py
.venv313\Scripts\python.exe scripts\audit_stage_c.py
.venv313\Scripts\python.exe scripts\check_doc_links.py
```

The `check_stage_*_complete.py` scripts ("completeness") check that expected
artifacts exist and are shaped correctly. `audit_stage_*.py` ("correctness")
independently re-derives every reported number from raw persisted JSON and
never trust a number back from the prose that reports it.
`check_doc_links.py` verifies every relative Markdown link resolves. All
nine of these run in CI on every push and PR (see
[`.github/workflows/docs-check.yml`](.github/workflows/docs-check.yml)).

## Ground rules

- **No silent patching.** If a reproduction diverges from a paper, the fix
  goes in a separate, clearly-labeled diagnostic path (see
  `scripts/diagnose_fedgraa.py`) rather than being folded quietly into the
  faithful reproduction. Document *why* it diverges before changing the code
  that reproduces it.
- **Every quantitative claim needs a source.** If a doc states a number, it
  should be traceable to a persisted result file or a cited external source,
  not just asserted.
- **Fix documentation drift as you find it.** If a link, a file path, or a
  claim in the docs no longer matches reality, fix it in the same PR rather
  than filing a follow-up.
- Keep diffs focused. Unrelated cleanup belongs in its own PR.

## Reporting a reproducibility issue

If you've tried to reproduce a result here and gotten a different number,
please open an issue with: your environment (OS, Python version, TensorFlow
version), the exact command you ran, and the full output. That's the single
most useful bug report this repo can receive.
