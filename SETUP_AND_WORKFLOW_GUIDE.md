# Setup & Workflow Guide — `nova-entity` environment

## 1. Environment setup in Anaconda Navigator

You already have a `nova-entity` environment from earlier work, so you're
mostly just refreshing packages, not starting over.

**Option A — Anaconda Navigator GUI:**
1. Open **Anaconda Navigator** → **Environments** (left sidebar).
2. Click **nova-entity** to select it.
3. Click the ▶ (play) icon next to it → **Open Terminal**. This opens a
   terminal with `nova-entity` already activated (you'll see `(nova-entity)`
   in the prompt).
4. In that terminal, `cd` to your project folder, e.g.:
   ```bash
   cd path\to\Nova-Submissions
   ```
5. Install/update the pinned dependencies:
   ```bash
   pip install -r code/business_entity_resolution/requirements.txt
   ```

**Option B — Anaconda Prompt (if you prefer typing over clicking):**
```bash
conda activate nova-entity
cd path\to\Nova-Submissions
pip install -r code/business_entity_resolution/requirements.txt
```

**Sanity check** (run this once, in the activated environment):
```bash
python -c "import pandas, numpy, rapidfuzz; print(pandas.__version__, numpy.__version__, rapidfuzz.__version__)"
```
You want to see `3.0.6 2.5.3 3.14.6` (or whatever `requirements.txt` says at
the time). If `pip install` grabbed different versions because of Python
version constraints, that's fine as long as this import line succeeds —
just note the actual versions in your `requirements.txt` before zipping the
final submission, since "pinned dependencies" is an explicit grading
criterion.

## 2. Drop in the renovated code

Replace your existing `code/business_entity_resolution/` folder's contents
with the files in this package's `code/business_entity_resolution/` folder
(or diff them in by hand if you've made local edits since your last upload).
Also copy `utils/validate_submission.py` from this package into your
project's `utils/` folder if it's not already there — it's identical to
the one you uploaded, just kept out of the submission-mirroring folder here.

Then clean out the dead weight per `ARCHIVE_NOTES.md`:
- Move `module1_baseline.py`, `module1_baseline-checkpoint.py`,
  `module2_fuzzy.py`, `module3_lsh.py`, `diagnose_*.py`,
  `evaluate_module2_train.py`, `evaluate_module3_train.py` into
  `code/experiments/` if they aren't already there (they're genuine R&D
  history, worth keeping for your own reference and for writing up
  methodology — just never in the submission zip).
- Delete the `*_cpython-312.pyc` files and `run_pipeline.py`.
- Delete the duplicate `matching_results_1.tsv` / `candidate_pairs_1.tsv`.

## 3. "Training" = threshold tuning against labeled data

There's no model with weights here, so training means: run the labeled
train set through the pipeline's own blocking+matching logic and read off
precision / recall / macro-F0.5, then adjust thresholds and repeat.

```bash
conda activate nova-entity
cd path\to\Nova-Submissions
python code/business_entity_resolution/src/evaluate.py --sample-size 10000
```

Read the output for:
- **Candidate Blocking Recall** — the ceiling on everything downstream. If
  this is low, no amount of matching-threshold tuning will fix it; you'd
  need to loosen blocking (bigger `DEFAULT_POOL_SIZE` in `blocking.py`, or a
  new key type).
- **Macro Precision / Recall / F0.5** — your actual score proxy.
- **Predicted entities with >10 matches** — the new sanity line. If this
  number is high relative to the printed "max true links" figure, tighten
  `MIN_INFORMATIVE_TOKENS` in `src/matching.py` or raise `MATCH_THRESHOLD`.

Things worth trying, one change at a time, re-running `evaluate.py` after
each to see the effect:
- `CANDIDATE_THRESHOLD` / `MATCH_THRESHOLD` in `src/matching.py`
- `MIN_INFORMATIVE_TOKENS` in `src/matching.py` (the new generic-name guard)
- `DEFAULT_POOL_SIZE` in `src/blocking.py` (bigger = higher recall ceiling,
  slower run)
- `max_bucket_size` on `CountryBlockingIndex` in `src/blocking.py`

Increase `--sample-size` (e.g. to 25000 or 50000) once you've narrowed in on
good values, to make sure they hold up on a bigger, less noisy sample —
your `evaluate.py` run should take at most a few minutes even at 50k.

## 4. "Testing" = running the real pipeline once, on the test set

Once you're happy with the thresholds:

```bash
python code/business_entity_resolution/run.py ^
    --data-dir dataset/test ^
    --output-dir output ^
    --cand-threshold 65.0 ^
    --match-threshold 88.0 ^
    --max-candidates 15 ^
    --max-matches 20 ^
    --pool-size 45
```
(Use `^` for line continuation on Windows cmd; `\` if you're in PowerShell/WSL/bash.)

This is the one and only run that produces your real
`output/matching_results.tsv` and `output/candidate_pairs.tsv`. Watch the
printed stage timings — Stage 1 (indexing Source 2+3) and Stage 2 (matching
Source 1) are reported separately, so if one stage dominates the runtime
you'll know where to focus if you want to optimize further.

## 5. Validate before you submit anything

```bash
python utils/validate_submission.py ^
    --matching output/matching_results.tsv ^
    --candidate output/candidate_pairs.tsv ^
    --test-dir dataset/test
```
Fix anything it reports as `FAIL` before uploading to the leaderboard. The
`--check-ids` flag is optional and memory-heavier; run it once near the end
if you want the extra confidence, not on every iteration.

## 6. Package the final submission zip

The required zip is **flatter** than your dev repo — it should contain
*only*:
```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── normalization.py
│       │   ├── blocking.py
│       │   ├── matching.py
│       │   ├── pipeline.py
│       │   └── evaluate.py
│       ├── run.py
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```
No `dataset/`, `notebooks/`, `utils/`, `code/experiments/`, or `.gitignore`
inside the zip. From your project root:

```bash
cd path\to\Nova-Submissions
python -c "
import shutil, os
os.makedirs('_submission_build/output', exist_ok=True)
os.makedirs('_submission_build/code', exist_ok=True)
shutil.copy('output/matching_results.tsv', '_submission_build/output/')
shutil.copy('output/candidate_pairs.tsv', '_submission_build/output/')
shutil.copytree('code/business_entity_resolution', '_submission_build/code/business_entity_resolution')
shutil.copy('Documentation_template.md', '_submission_build/')
shutil.make_archive('team_name_submission', 'zip', '_submission_build')
print('Built team_name_submission.zip')
"
```
Rename `team_name_submission.zip` to your actual team name before uploading,
and run `validate_submission.py` one more time against the *unzipped*
contents of `_submission_build/output/` as a final check.

## 7. Update `Documentation_template.md` before zipping

Your existing draft (uploaded) is well-written and mostly still accurate —
the architecture and normalization/blocking sections describe the code
correctly. Before final submission, update:
- **Section 6 (Empirical Validation table)** — re-run `evaluate.py` with the
  renovated code and paste in the *new* numbers; the old table reflects the
  pre-renovation matcher.
- **Section 2 diagram / Section 5** — optionally mention the generic-name
  guard and the vectorized scoring change, since "any other relevant
  information about the approach" is explicitly requested and this is a
  genuine, explainable design decision.
