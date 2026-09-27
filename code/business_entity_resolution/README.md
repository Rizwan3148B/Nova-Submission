# Business Entity Resolution Pipeline

End-to-end pipeline for resolving business entities across Source 1
(deduplicated reference), Source 2, and Source 3 records, scored by
macro-averaged F0.5. See `CHANGELOG.md` for what changed in this renovation
pass versus the previous production version.

## Overview & Architecture

Three stages:

1. **Multilingual Normalization Engine (`src/normalization.py`)**
   Unicode NFKD ASCII folding, legal-suffix/stopword filtering, address
   canonicalization, postal-code extraction, and country-key normalization.

2. **Country-Partitioned Inverted Index Blocking (`src/blocking.py`)**
   One inverted index per (normalized) country. Multi-key union of an exact
   normalized-name key, a token-pair-prefix key, individual token-prefix
   keys, and a postal-anchored key. Pathological buckets are capped, not
   dropped.

3. **RapidFuzz Composite Matcher (`src/matching.py`)**
   Vectorized `token_sort_ratio` + `token_set_ratio` name scoring, weighted
   0.70 name / 0.30 address, postal bonus/penalty, a generic-name guard, and
   an explicit safety cap on matches per entity.

```text
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── normalization.py
│   ├── blocking.py
│   ├── matching.py
│   ├── pipeline.py
│   └── evaluate.py
├── run.py
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## "Training" (threshold tuning on labeled train data)

There's no model to fit here — tuning means running the labeled train set
through the same blocking+matching logic and reading off precision/recall/F0.5:

```bash
python code/business_entity_resolution/src/evaluate.py --sample-size 10000
```

Adjust `CANDIDATE_THRESHOLD`, `MATCH_THRESHOLD`, `MIN_INFORMATIVE_TOKENS` in
`src/matching.py` between runs until the numbers look right, then move to
the real test set.

## Running on the test set

```bash
python code/business_entity_resolution/run.py \
    --data-dir dataset/test \
    --output-dir output \
    --cand-threshold 65.0 \
    --match-threshold 88.0 \
    --max-candidates 15 \
    --max-matches 20 \
    --pool-size 45
```

Produces:
- `output/matching_results.tsv` — leaderboard submission file
- `output/candidate_pairs.tsv` — blocking candidate set fed to the matcher

## Validating before submission

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
