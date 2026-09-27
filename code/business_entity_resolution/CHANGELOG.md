# Changelog — Renovation Pass

Everything here is a change to the *production* pipeline you already had
working (`src/normalization.py`, `blocking.py`, `matching.py`, `pipeline.py`,
`evaluate.py`, `run.py`). Nothing in `code/experiments/` was touched or
carried forward — see `ARCHIVE_NOTES.md` for what happened to those files.

## 1. Correctness: country-key normalization
**Files:** `normalization.py` (new `clean_country`), `pipeline.py`, `evaluate.py`

The old code partitioned every index and lookup by the raw `country`
string. Source 2/3 indexing did `str(country_val)`; Source 1 lookup did the
same, separately. If the two files ever disagreed on casing or had a stray
space, a Source 1 record would look up a country key with no index, silently
get zero candidates, and come out as a false singleton — with no error
anywhere to reveal it. `clean_country()` (`strip().lower()`) is now the only
way a country value is ever used as a key, in both places.

Cheap to add, can only ever recover recall, never hurts precision.

## 2. Performance: vectorized name scoring
**File:** `matching.py`

The old `score_pair()` called `fuzz.token_sort_ratio` and
`fuzz.token_set_ratio` once each, per candidate, in a Python loop — up to
~45-50 times per Source 1 record. `score_and_rank_candidates()` now calls
`rapidfuzz.process.cdist()` once per metric per record, scoring the whole
candidate pool in a single C-level loop. The per-candidate math (early-exit
thresholds, address scoring, postal bonus/penalty) is unchanged — this only
changes *how* the name-similarity numbers get computed, not what they mean.
Expect a real wall-clock improvement on the full ~1.7M-row test set; exact
speedup depends on your machine, so time both versions on a 50k-row slice if
you want a number.

## 3. Precision: generic-name guard
**Files:** `normalization.py` (new `count_informative_tokens`), `matching.py`

Diagnosed from your own `candidate_pairs.tsv` / `matching_results.tsv`:
sampling showed some Source 1 entities matched to **30–49** records, while
your training ground truth's observed max is **11**. That's the signature of
a generic/short business name (after legal-suffix stripping, e.g. "sri
traders") clearing the similarity bar against many unrelated real
businesses in the same bucket.

A name with fewer than `MIN_INFORMATIVE_TOKENS` (default 1) non-stopword
tokens now:
- cannot use the 100.0 exact-match fast path, and
- cannot reach `MATCH_THRESHOLD` on name+address similarity alone — it also
  needs a matching postal code.

This is a starting point, not a proven-optimal cutoff. Run `evaluate.py`
before and after to see whether it moves precision the way you'd expect on
your own labeled sample, and adjust `MIN_INFORMATIVE_TOKENS` if it's too
aggressive or not aggressive enough.

## 4. Robustness: explicit caps, wired through
**Files:** `blocking.py`, `matching.py`, `pipeline.py`, `run.py`

- `blocking.get_candidates()`'s retrieval pool size is now the named
  constant `DEFAULT_POOL_SIZE` (45), importable and passed explicitly by
  `pipeline.py`/`evaluate.py`, instead of a bare `50` default that
  `pipeline.py` was silently relying on.
- `matching.MAX_MATCHES_PER_S1` (default 20) is a new, explicit ceiling on
  matches per Source 1 entity — comfortably above your training ground
  truth's observed max of 11, so a bad bucket is now bounded and visible
  (`evaluate.py` prints how many predicted entities exceed 10 matches) rather
  than open-ended.
- `run.py` exposes `--max-matches` and `--pool-size` as CLI flags alongside
  the existing threshold flags, so the whole cap chain
  (pool → candidates → matches) is tunable without editing source.

## What did NOT change
- The overall 3-stage architecture (normalize → country-partitioned
  multi-key blocking → RapidFuzz composite scoring) is unchanged — it's a
  reasonable design and the empirical numbers in your own
  `Documentation_template.md` (0.864 blocking recall, 0.887 macro-F0.5 on
  your evaluation sample) suggest it was already working.
- `CANDIDATE_THRESHOLD` (65.0), `MATCH_THRESHOLD` (88.0), `NAME_WEIGHT`/
  `ADDRESS_WEIGHT` (0.70/0.30), and `MAX_CANDIDATES_PER_S1` (15) are
  unchanged. Re-tune these with `evaluate.py` if the generic-name guard
  shifts your precision/recall balance enough to want a different threshold.
