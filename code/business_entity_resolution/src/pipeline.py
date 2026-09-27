"""
Production End-to-End Pipeline for Business Entity Resolution.
Generates output/matching_results.tsv and output/candidate_pairs.tsv.
Optimized single-pass inverted index architecture across US, India, and France.

CHANGES vs the original module (see CHANGELOG.md at the package root):
  - Every place that reads the `country` column now goes through
    normalization.clean_country() before it's used as a dict key or
    compared. In the original code, Source 2/3 indexing used
    `str(country_val)` verbatim (whatever casing/whitespace the raw file
    had) while Source 1 lookup separately re-cast with `str(...)`. If the
    two files ever disagreed on casing or had incidental whitespace, a
    Source 1 record would silently find country_indexes.get(...) returning
    None and be forced into a false singleton -- with no error, warning, or
    row in either output file to reveal it happened. This is now
    structurally impossible for that specific cause.
  - get_candidates() is called with an explicit pool_size (imported from
    blocking.DEFAULT_POOL_SIZE) instead of relying on that function's own
    default, and score_and_rank_candidates() is called with explicit
    max_matches (from matching.MAX_MATCHES_PER_S1) so the whole cap chain
    (blocking pool -> candidate cap -> match cap) is visible in one place.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import gc
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from .normalization import clean_name, clean_address, clean_country, extract_postal_code
from .blocking import CountryBlockingIndex, DEFAULT_POOL_SIZE
from .matching import (
    score_and_rank_candidates,
    CANDIDATE_THRESHOLD,
    MATCH_THRESHOLD,
    MAX_CANDIDATES_PER_S1,
    MAX_MATCHES_PER_S1,
)

CHUNK_SIZE = 100_000
S1_CHUNK_SIZE = 50_000


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    cand_threshold: float = CANDIDATE_THRESHOLD,
    match_threshold: float = MATCH_THRESHOLD,
    max_candidates: int = MAX_CANDIDATES_PER_S1,
    max_matches: int = MAX_MATCHES_PER_S1,
    pool_size: int = DEFAULT_POOL_SIZE,
):
    start_total = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    s1_path = data_dir / "test_source1.tsv"
    s2_path = data_dir / "test_source2.tsv"
    s3_path = data_dir / "test_source3.tsv"

    for p in [s1_path, s2_path, s3_path]:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION PIPELINE (renovated)")
    print("=" * 70)
    print(f"Data directory      : {data_dir}")
    print(f"Output directory    : {output_dir}")
    print(f"Candidate threshold : {cand_threshold}")
    print(f"Match threshold     : {match_threshold}")
    print(f"Max candidates / S1 : {max_candidates}")
    print(f"Max matches / S1    : {max_matches}")
    print(f"Blocking pool size  : {pool_size}")

    # Remove existing outputs
    for p in [matching_path, candidate_path]:
        if p.exists():
            p.unlink()

    # 1. Build Country Inverted Indexes in a single unified pass over S2 and S3
    print("\n" + "=" * 70)
    print("STAGE 1: BUILDING MULTI-KEY BLOCKING INDEXES (S2 + S3)")
    print("=" * 70)

    # Dictionary mapping normalized-country -> CountryBlockingIndex
    country_indexes: Dict[str, CountryBlockingIndex] = {}

    idx_start = time.time()
    for s_path, s_name in [(s2_path, "Source 2"), (s3_path, "Source 3")]:
        print(f"\nScanning and indexing {s_name} ({s_path.name})...")
        chunk_count = 0
        s_records = 0

        for chunk in pd.read_csv(
            s_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            usecols=["entity_id", "business_name", "business_address", "country"]
        ):
            chunk_count += 1
            s_records += len(chunk)

            # Normalize country BEFORE grouping so casing/whitespace never
            # fragments a country into two silent buckets.
            chunk["country_key"] = chunk["country"].map(clean_country)

            for country_key, c_group in chunk.groupby("country_key", sort=False):
                if not country_key:
                    continue
                if country_key not in country_indexes:
                    country_indexes[country_key] = CountryBlockingIndex(country_key)

                c_names = [clean_name(n) for n in c_group["business_name"]]
                c_addrs = [clean_address(a) for a in c_group["business_address"]]
                country_indexes[country_key].add_records(c_group["entity_id"].tolist(), c_names, c_addrs)

            if chunk_count % 10 == 0:
                elapsed = time.time() - idx_start
                print(f"  {s_name}: chunk {chunk_count:,} ({s_records:,} rows) | elapsed: {elapsed/60:.1f} min")

    print(f"\nFinished indexing all sources in {(time.time() - idx_start)/60:.2f} minutes.")
    print("Index statistics by country:")
    for c_key, c_idx in sorted(country_indexes.items()):
        capped = c_idx.prune_large_buckets()
        print(f"  [{c_key}]: {c_idx.num_records:,} records | {len(c_idx.index):,} active keys ({capped} buckets capped)")

    gc.collect()

    # 2. Process Source 1 in streaming chunks and directly write final outputs
    print("\n" + "=" * 70)
    print("STAGE 2: STREAMING SOURCE 1 MATCHING & CANDIDATE GENERATION")
    print("=" * 70)

    total_s1 = 0
    total_matched = 0
    total_singletons = 0
    total_cand_rows = 0
    total_no_country_index = 0

    first_chunk = True
    match_start = time.time()
    s1_chunk_num = 0

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        chunksize=S1_CHUNK_SIZE,
        usecols=["entity_id", "business_name", "business_address", "country"]
    ):
        s1_chunk_num += 1
        chunk_rows = len(chunk)
        total_s1 += chunk_rows

        c_names = [clean_name(n) for n in chunk["business_name"]]
        c_addrs = [clean_address(a) for a in chunk["business_address"]]
        c_ids = chunk["entity_id"].tolist()
        c_countries = [clean_country(c) for c in chunk["country"]]

        match_rows = []
        cand_rows = []

        chunk_matched = 0
        chunk_candidates = 0

        query_cache = {}

        for i in range(chunk_rows):
            s1_id = str(c_ids[i])
            country_key = c_countries[i]
            s1_name = c_names[i]
            s1_addr = c_addrs[i]
            s1_postal = extract_postal_code(s1_addr)

            c_idx = country_indexes.get(country_key)
            if not c_idx or not s1_name:
                if country_key and not c_idx:
                    total_no_country_index += 1
                match_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ""})
                cand_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ""})
                continue

            cache_key = (country_key, s1_name, s1_addr)
            if cache_key in query_cache:
                cand_ids, match_ids = query_cache[cache_key]
            else:
                cand_indices = c_idx.get_candidates(s1_name, s1_postal, pool_size=pool_size)
                cand_ids, match_ids = score_and_rank_candidates(
                    s1_name, s1_addr, s1_postal, cand_indices, c_idx,
                    cand_threshold=cand_threshold,
                    match_threshold=match_threshold,
                    max_candidates=max_candidates,
                    max_matches=max_matches,
                )
                query_cache[cache_key] = (cand_ids, match_ids)

            match_rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(match_ids)
            })

            cand_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(cand_ids)
            })

            if match_ids:
                chunk_matched += 1
            if cand_ids:
                chunk_candidates += 1

        total_matched += chunk_matched
        total_singletons += (chunk_rows - chunk_matched)
        total_cand_rows += chunk_candidates

        m_df = pd.DataFrame(match_rows)
        c_df = pd.DataFrame(cand_rows)

        m_df.to_csv(
            matching_path, sep="\t", index=False,
            mode="w" if first_chunk else "a", header=first_chunk
        )
        c_df.to_csv(
            candidate_path, sep="\t", index=False,
            mode="w" if first_chunk else "a", header=first_chunk
        )

        first_chunk = False

        if s1_chunk_num % 5 == 0 or total_s1 == chunk_rows:
            elapsed = time.time() - match_start
            coverage = 100.0 * total_matched / total_s1 if total_s1 else 0.0
            print(
                f"Source 1: processed {total_s1:,} rows | "
                f"matched: {total_matched:,} ({coverage:.1f}%) | "
                f"singletons: {total_singletons:,} | "
                f"no-country-index: {total_no_country_index:,} | "
                f"elapsed: {elapsed/60:.1f} min"
            )

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED")
    print("=" * 70)
    print(f"Total Source 1 entities processed : {total_s1:,}")
    print(f"Entities with predicted matches   : {total_matched:,}")
    print(f"Singletons (no match)             : {total_singletons:,} ({total_singletons/total_s1:.2%})")
    print(f"Entities with candidates          : {total_cand_rows:,} ({total_cand_rows/total_s1:.2%})")
    if total_no_country_index:
        print(
            f"WARNING: {total_no_country_index:,} Source 1 rows had a country "
            f"value with no matching Source 2/3 index at all (forced singleton). "
            f"Inspect these -- this usually means a country label appears only "
            f"in Source 1, which is expected for some rows but worth a spot check."
        )
    print(f"Total Execution Time              : {total_time / 60:.2f} minutes")
    print("\nGenerated files:")
    print(f"  {matching_path} ({os.path.getsize(matching_path)/(1024*1024):.1f} MB)")
    print(f"  {candidate_path} ({os.path.getsize(candidate_path)/(1024*1024):.1f} MB)")


if __name__ == "__main__":
    proj_dir = Path(__file__).resolve().parents[3]
    d_dir = proj_dir / "dataset" / "test"
    o_dir = proj_dir / "output"
    run_pipeline(d_dir, o_dir)
