# File Audit — What Every File Does, and What To Do With It

This covers everything you've uploaded across both messages. Verdicts:
**KEEP** = renovated version is in this package, use it. **ARCHIVE** = real
R&D history, worth keeping in `code/experiments/` in your own repo for your
own reference, but never goes in the submission zip. **DELETE** = safe to
throw away entirely — either a stale duplicate, a compiled artifact, or an
empty file.

## Production pipeline (KEEP — renovated versions in this package)

| File | Role |
|---|---|
| `src/__init__.py` | Package marker. Unchanged. |
| `src/normalization.py` | Name/address/country cleaning, postal extraction. Renovated: added `clean_country`, `count_informative_tokens`. |
| `src/blocking.py` | Country-partitioned multi-key inverted index (the candidate generator). Renovated: named `DEFAULT_POOL_SIZE` instead of a hidden default. |
| `src/matching.py` | RapidFuzz composite scorer + threshold split into candidates/matches. Renovated: vectorized scoring, generic-name guard, `MAX_MATCHES_PER_S1` safety cap. |
| `src/pipeline.py` | Full streaming run over the test set → `matching_results.tsv` + `candidate_pairs.tsv`. Renovated: consistent country-key normalization, explicit cap wiring. |
| `src/evaluate.py` | Local scorer against train ground truth (your "training" step). Renovated: same country fix, sample size as a CLI flag, prints a match-count sanity check. |
| `run.py` | CLI entrypoint. Renovated: exposes `--max-matches` / `--pool-size`. |
| `requirements.txt`, `README.md` | Kept, README updated for the new flags. |

## R&D history (ARCHIVE — do not put in the submission zip)

These were your genuine path to the current design, and the diagnostic
scripts already told you real things (e.g. first-token-only blocking misses
~27% of true links — that's *why* `blocking.py`'s multi-key approach exists).
Worth keeping for your own record and for writing the methodology doc, but
none of them should ship in `code/business_entity_resolution/`.

| File | What it was for |
|---|---|
| `module1_baseline.py` | Exact-normalized-name-only matching. First working baseline; superseded entirely by the fuzzy pipeline. |
| `module1_baseline-checkpoint.py` | A Jupyter notebook (despite the `.py` name) containing the same Module 1 code. Its saved output shows a `NameError: __file__ is not defined` — harmless, just what happens when `Path(__file__)` runs inside a notebook cell instead of a script. Not worth fixing since Module 1 is retired. |
| `module2_fuzzy.py` | First-token bucketing (with a second-token-prefix split for oversized buckets) + RapidFuzz scoring. This is the version your diagnostics measured at 0.5758 blocking recall. Superseded by `blocking.py`'s multi-key index. |
| `module3_lsh.py` | Same thresholds (65/88) and the same "informative token" blocking-key idea as the current production `blocking.py` — this is essentially the prototype that became `src/blocking.py` + `src/matching.py` + `src/pipeline.py` combined into one file. Fully superseded. |
| `diagnose_blocking.py` | Measured Module 2's first-token / second-prefix / exact-name recall on a 25k sample. Result: first-token recall only ~73% (27% of true links have a different first token) — this is the evidence base for building a multi-key index. |
| `diagnose_actual_blocking.py` | Same idea, closer trace of Module 2's actual bucket-size-dependent logic (small buckets = first-token only, large buckets = first token + second-token prefix). |
| `diagnose_candidate_loss.py` | Asks a sharper question: of the true links that *do* reach a block, where does the true match rank once RapidFuzz scores the whole bucket? Useful methodology, not needed as a script going forward unless you want to re-run this kind of rank analysis on the new blocking. |
| `evaluate_module2_train.py` | Sanity-eval harness for Module 2 specifically (imports from `module2_fuzzy.py`). Superseded by `src/evaluate.py`, which does the same job against the current production code. |
| `evaluate_module3_train.py` | Same idea for Module 3. Superseded likewise. |
| `01_setup_and_dataset_check.ipynb` | Notebook, presumably early data exploration. Keep in `notebooks/` for your own reference; irrelevant to the submission. |
| `01_setup_and_dataset_check-checkpoint.ipynb` | An empty Jupyter autosave checkpoint (`"cells": []`). |

## Safe to delete outright

| File | Why |
|---|---|
| `*_cpython-312.pyc` (all seven) | Compiled bytecode caches (`__pycache__` contents). Regenerated automatically; never belong in a repo or a zip. |
| `run_pipeline.py` | Thin wrapper that just calls `run.py`'s `main()`. Redundant — `run.py` is already the documented entrypoint. Delete unless you specifically use this shortcut. |
| `matching_results_1.tsv`, `candidate_pairs_1.tsv` | Byte-for-byte identical duplicates of `matching_results.tsv` / `candidate_pairs.tsv` (verified via checksum). Almost certainly an accidental double-upload/copy. |
| `01_setup_and_dataset_check-checkpoint.ipynb` | Empty notebook checkpoint, see above. |

## Data files (informational — these are outputs, not code)

| File | What it showed |
|---|---|
| `matching_results.tsv` / `candidate_pairs.tsv` (the 1.73M-row pair) | Your last full test-set run. **Stale relative to the renovated code** — re-run `run.py` before submitting. Diagnosis: avg candidates/entity was 18.1 (max 50) and avg matches/entity 2.26 with some entities matched to 30-49 records — see `CHANGELOG.md` items 3-4 for the fixes this drove. |
| `train_matching_results.tsv` / `train_candidate_pairs.tsv` (5,000-row pair) | Output of an `evaluate.py` run on a 5k train sample. Avg candidates 9.8 (max 18), avg matches 2.24, 12.9% predicted singletons. Re-generate with the renovated `evaluate.py` for a fresh baseline. |
| `module2_blocking_diagnostic.tsv` | Per-true-link diagnostic rows from `diagnose_blocking.py`'s 25k sample. Confirms the 27%-recall-loss finding above. Reference only. |
