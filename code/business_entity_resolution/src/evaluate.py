"""
Comprehensive Train-Set Evaluation for Business Entity Resolution.
Evaluates Blocking Recall, Match Precision, Recall, and official Macro-F0.5.

This is your "training" step for this challenge: there is no model to fit
gradients for, so "training" means using labeled train data to measure and
tune CANDIDATE_THRESHOLD / MATCH_THRESHOLD / MAX_CANDIDATES_PER_S1 in
matching.py before you ever touch the test set. Run this, look at the
numbers, adjust thresholds, run again -- then run the real pipeline once.

CHANGES vs the original module: same clean_country() fix as pipeline.py,
plus the sample size is now a CLI argument instead of a hardcoded number at
the bottom of the file (see __main__).
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Tuple
import pandas as pd

curr_dir = Path(__file__).resolve().parent.parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

try:
    from .normalization import clean_name, clean_address, clean_country, extract_postal_code
    from .blocking import CountryBlockingIndex, DEFAULT_POOL_SIZE
    from .matching import (
        score_and_rank_candidates,
        CANDIDATE_THRESHOLD,
        MATCH_THRESHOLD,
        MAX_CANDIDATES_PER_S1,
        MAX_MATCHES_PER_S1,
    )
except (ImportError, ValueError):
    from src.normalization import clean_name, clean_address, clean_country, extract_postal_code
    from src.blocking import CountryBlockingIndex, DEFAULT_POOL_SIZE
    from src.matching import (
        score_and_rank_candidates,
        CANDIDATE_THRESHOLD,
        MATCH_THRESHOLD,
        MAX_CANDIDATES_PER_S1,
        MAX_MATCHES_PER_S1,
    )

PROJECT_DIR = Path(__file__).resolve().parents[3]
TRAIN_DIR = PROJECT_DIR / "dataset" / "train"

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"


def compute_macro_f05(ground_truth_map: dict, predictions_map: dict) -> Tuple[float, float, float]:
    """
    Compute official macro-averaged F0.5 per Source 1 entity across the evaluation set,
    including singletons.
    """
    total_f05 = 0.0
    total_p = 0.0
    total_r = 0.0
    n = len(ground_truth_map)

    if n == 0:
        return 0.0, 0.0, 0.0

    beta = 0.5
    beta_sq = beta ** 2

    for s1_id, true_set in ground_truth_map.items():
        pred_set = set(predictions_map.get(s1_id, []))

        if len(true_set) == 0:
            if len(pred_set) == 0:
                entity_f05 = entity_p = entity_r = 1.0
            else:
                entity_f05 = entity_p = entity_r = 0.0
        else:
            tp = len(pred_set & true_set)
            p = tp / len(pred_set) if len(pred_set) > 0 else 0.0
            r = tp / len(true_set)

            if p + r > 0:
                entity_f05 = (1.0 + beta_sq) * p * r / (beta_sq * p + r)
            else:
                entity_f05 = 0.0

            entity_p = p
            entity_r = r

        total_f05 += entity_f05
        total_p += entity_p
        total_r += entity_r

    return total_f05 / n, total_p / n, total_r / n


def run_evaluation(sample_size: int = 10000):
    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION — VALIDATION & EVALUATION (renovated)")
    print("=" * 70)

    print(f"Loading {sample_size:,} Source 1 training records...")
    s1_df = pd.read_csv(TRAIN_S1, sep="\t", nrows=sample_size * 2, dtype=str).dropna(subset=["entity_id"])
    s1_df = s1_df.sample(n=min(sample_size, len(s1_df)), random_state=42).reset_index(drop=True)

    s1_ids = set(s1_df["entity_id"].astype(str))

    print("Loading matching ground truth...")
    gt_map = {}
    total_true_links = 0

    for chunk in pd.read_csv(GROUND_TRUTH, sep="\t", chunksize=100_000, dtype=str):
        chunk = chunk.fillna("")
        id_col = "source1_entity_id" if "source1_entity_id" in chunk.columns else "entity_id"
        rel = chunk[chunk[id_col].isin(s1_ids)]
        for _, row in rel.iterrows():
            s1_id = str(row[id_col])
            val = str(row["matched_entity_ids"]).strip()
            if val:
                t_ids = set(x.strip() for x in val.split(",") if x.strip())
                gt_map[s1_id] = t_ids
                total_true_links += len(t_ids)
            else:
                gt_map[s1_id] = set()
        if len(gt_map) == len(s1_ids):
            break

    for s1_id in s1_ids:
        if s1_id not in gt_map:
            gt_map[s1_id] = set()

    singletons_count = sum(1 for s in gt_map.values() if len(s) == 0)
    print(f"Evaluated S1 entities: {len(gt_map):,} (Singletons: {singletons_count:,} / {singletons_count/len(gt_map):.2%})")
    print(f"Total true link edges: {total_true_links:,}")
    if gt_map:
        max_links = max(len(s) for s in gt_map.values())
        print(f"Max true links for a single S1 entity: {max_links}  <- use this to sanity-check MAX_MATCHES_PER_S1={MAX_MATCHES_PER_S1}")

    print("\nBuilding Country Blocking Indexes from Train Source 2 & Source 3...")
    t0 = time.time()
    country_indexes = {}

    for src_path, label in [(TRAIN_S2, "Source 2"), (TRAIN_S3, "Source 3")]:
        print(f"  Indexing {label}...")
        chunk_num = 0
        for chunk in pd.read_csv(src_path, sep="\t", chunksize=100_000, dtype=str):
            chunk_num += 1
            chunk["country_key"] = chunk["country"].map(clean_country)
            for country_key, c_group in chunk.groupby("country_key", sort=False):
                if not country_key:
                    continue
                if country_key not in country_indexes:
                    country_indexes[country_key] = CountryBlockingIndex(country_key)
                clean_names = [clean_name(n) for n in c_group["business_name"]]
                clean_addrs = [clean_address(a) for a in c_group["business_address"]]
                country_indexes[country_key].add_records(
                    c_group["entity_id"].tolist(), clean_names, clean_addrs
                )
            if chunk_num % 10 == 0:
                print(f"    {label}: processed chunk {chunk_num} ({chunk_num*100_000:,} rows)...")

    for ctry, c_idx in country_indexes.items():
        capped = c_idx.prune_large_buckets()
        print(f"  [{ctry}] Indexed records: {c_idx.num_records:,}, Keys: {len(c_idx.index):,}, Capped buckets: {capped}")

    print(f"Indexing completed in {(time.time() - t0)/60:.2f} minutes.")

    print("\nRunning Matching on Source 1 sample...")
    cand_preds = {}
    match_preds = {}

    # IMPORTANT: keep two different recall measurements separate.
    # 1) raw_block_hits: true links retrieved by blocking BEFORE scoring.
    # 2) post_score_cand_hits: true links that survive scoring / candidate threshold.
    # The previous evaluator called (2) "Candidate Blocking Recall", which
    # mixed blocking quality with matching/scoring behavior.
    raw_block_hits = 0
    post_score_cand_hits = 0
    no_country_index = 0

    t1 = time.time()
    for _, row in s1_df.iterrows():
        s1_id = str(row["entity_id"])
        ctry = clean_country(str(row["country"]))
        s1_name = clean_name(row["business_name"])
        s1_addr = clean_address(row["business_address"])
        s1_postal = extract_postal_code(s1_addr)

        true_targets = gt_map.get(s1_id, set())

        c_idx = country_indexes.get(ctry)
        if not c_idx or not s1_name:
            if ctry and not c_idx:
                no_country_index += 1
            cand_preds[s1_id] = []
            match_preds[s1_id] = []
            continue

        cand_indices = c_idx.get_candidates(
            s1_name, s1_postal, pool_size=DEFAULT_POOL_SIZE
        )

        # Pure blocking recall: inspect the raw indices returned by the blocker
        # BEFORE any fuzzy scoring or candidate threshold is applied.
        raw_cand_ids = {c_idx.entity_ids[i] for i in cand_indices}
        raw_block_hits += len(raw_cand_ids & true_targets)

        cand_ids, match_ids = score_and_rank_candidates(
            s1_name, s1_addr, s1_postal, cand_indices, c_idx
        )

        cand_preds[s1_id] = cand_ids
        match_preds[s1_id] = match_ids

        # This is useful too, but it is NOT pure blocking recall.
        post_score_cand_hits += len(set(cand_ids) & true_targets)

    eval_time = time.time() - t1
    print(f"Matching finished in {eval_time:.1f}s ({len(s1_df)/eval_time:.1f} entities/sec).")
    if no_country_index:
        print(f"NOTE: {no_country_index:,} sample rows had a country with no train S2/S3 index at all.")

    macro_f05, macro_p, macro_r = compute_macro_f05(gt_map, match_preds)
    raw_blocking_recall = (
        raw_block_hits / total_true_links if total_true_links > 0 else 0.0
    )
    post_score_candidate_recall = (
        post_score_cand_hits / total_true_links if total_true_links > 0 else 0.0
    )

    cand_counts = [len(c) for c in cand_preds.values()]
    avg_cand_count = sum(cand_counts) / len(cand_counts) if cand_counts else 0.0

    match_counts = [len(m) for m in match_preds.values()]
    max_match_count = max(match_counts) if match_counts else 0
    over_10 = sum(1 for c in match_counts if c > 10)

    eval_out_dir = PROJECT_DIR / "output" / "train_eval"
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    m_rows = [{"source1_entity_id": s1, "matched_entity_ids": ",".join(m)} for s1, m in match_preds.items()]
    c_rows = [{"source1_entity_id": s1, "candidate_entity_ids": ",".join(c)} for s1, c in cand_preds.items()]

    pd.DataFrame(m_rows).to_csv(eval_out_dir / "train_matching_results.tsv", sep="\t", index=False)
    pd.DataFrame(c_rows).to_csv(eval_out_dir / "train_candidate_pairs.tsv", sep="\t", index=False)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Raw Blocking Recall (pre-score)      : {raw_blocking_recall:.4f}")
    print(f"Post-score Candidate Recall          : {post_score_candidate_recall:.4f}")
    print(f"Average Candidates per Entity        : {avg_cand_count:.2f}")
    print(f"Macro Precision (P)           : {macro_p:.4f}")
    print(f"Macro Recall (R)              : {macro_r:.4f}")
    print(f"Official Macro-F0.5 Score     : {macro_f05:.4f}")
    print(f"Predicted entities with >10 matches : {over_10:,}  (max: {max_match_count})")
    if over_10:
        print(
            "  -> If this number looks high relative to your true max link count above, "
            "the generic-name guard's MIN_INFORMATIVE_TOKENS in matching.py may need "
            "raising, or MATCH_THRESHOLD needs to go up."
        )
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local train-set evaluation / threshold tuning.")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of Source 1 training rows to sample.")
    args = parser.parse_args()
    run_evaluation(sample_size=args.sample_size)
