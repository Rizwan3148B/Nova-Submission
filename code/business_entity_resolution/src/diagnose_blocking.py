"""
Diagnose why true training links are missing from the renovated blocking stage.

This is a development diagnostic, NOT part of the final submission.

It measures:
  - country-index misses
  - which blocking key families can retrieve true links
  - how many true links are present in retained key buckets but lost by pool-size/early-return behavior
  - likely token-bucket pruning losses
  - raw blocker recall using the current implementation

Run from the Nova-Submission repository root:
    python code/business_entity_resolution/src/diagnose_blocking.py --sample-size 1000
"""

import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# Make package imports work when called as a script.
PACKAGE_DIR = Path(__file__).resolve().parent.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from src.normalization import clean_name, clean_address, clean_country, extract_postal_code
from src.blocking import CountryBlockingIndex, DEFAULT_POOL_SIZE

PROJECT_DIR = Path(__file__).resolve().parents[3]
TRAIN_DIR = PROJECT_DIR / "dataset" / "train"

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"


def load_sample_and_ground_truth(sample_size: int):
    print(f"Loading {sample_size:,} Source 1 training rows...")
    s1_df = (
        pd.read_csv(TRAIN_S1, sep="\t", nrows=sample_size * 2, dtype=str)
        .dropna(subset=["entity_id"])
    )
    s1_df = s1_df.sample(
        n=min(sample_size, len(s1_df)),
        random_state=42
    ).reset_index(drop=True)

    s1_ids = set(s1_df["entity_id"].astype(str))

    gt_map = {}
    total_true_links = 0

    print("Loading ground truth rows relevant to the sample...")
    for chunk in pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        chunksize=100_000,
        dtype=str,
    ):
        chunk = chunk.fillna("")
        id_col = "source1_entity_id" if "source1_entity_id" in chunk.columns else "entity_id"

        rel = chunk[chunk[id_col].isin(s1_ids)]

        for _, row in rel.iterrows():
            s1_id = str(row[id_col])
            val = str(row["matched_entity_ids"]).strip()

            if val:
                ids = {x.strip() for x in val.split(",") if x.strip()}
            else:
                ids = set()

            gt_map[s1_id] = ids

        if len(gt_map) == len(s1_ids):
            break

    for s1_id in s1_ids:
        gt_map.setdefault(s1_id, set())

    total_true_links = sum(len(v) for v in gt_map.values())

    print(f"Sample entities : {len(s1_df):,}")
    print(f"True link edges : {total_true_links:,}")

    return s1_df, gt_map


def build_indexes_and_target_locations(gt_map):
    # Only keep mappings for ground-truth target IDs from this sample.
    target_ids = set()
    for ids in gt_map.values():
        target_ids.update(ids)

    country_indexes = {}
    target_locations = {}
    t0 = time.time()

    for src_path, label in [(TRAIN_S2, "Source 2"), (TRAIN_S3, "Source 3")]:
        print(f"\nIndexing {label}...")

        chunk_num = 0

        for chunk in pd.read_csv(
            src_path,
            sep="\t",
            chunksize=100_000,
            dtype=str,
        ):
            chunk_num += 1
            chunk = chunk.fillna("")
            chunk["country_key"] = chunk["country"].map(clean_country)

            for country_key, c_group in chunk.groupby("country_key", sort=False):
                if not country_key:
                    continue

                if country_key not in country_indexes:
                    country_indexes[country_key] = CountryBlockingIndex(country_key)

                c_idx = country_indexes[country_key]

                names = [clean_name(x) for x in c_group["business_name"]]
                addrs = [clean_address(x) for x in c_group["business_address"]]

                base_idx = c_idx.num_records
                entity_ids = c_group["entity_id"].tolist()

                c_idx.add_records(entity_ids, names, addrs)

                # Save only locations that occur in this sample's ground truth.
                for offset, eid in enumerate(entity_ids):
                    eid = str(eid)
                    if eid in target_ids:
                        target_locations[eid] = (country_key, base_idx + offset)

            if chunk_num % 10 == 0:
                print(f"  {label}: processed {chunk_num} chunks ({chunk_num * 100_000:,} rows)...")

    print(f"\nIndexing completed in {(time.time() - t0) / 60:.2f} minutes.")

    for country_key, c_idx in sorted(country_indexes.items()):
        capped = c_idx.prune_large_buckets()
        print(
            f"  [{country_key}] records={c_idx.num_records:,}, "
            f"keys={len(c_idx.index):,}, capped_token_buckets={capped:,}"
        )

    missing_targets = target_ids - set(target_locations)
    if missing_targets:
        print(
            f"WARNING: {len(missing_targets):,} ground-truth target IDs were "
            f"not found in Source 2/3 while building indexes."
        )

    return country_indexes, target_locations


def key_family(key: str) -> str:
    if key.startswith("E:"):
        return "exact"
    if key.startswith("P:"):
        return "pair"
    if key.startswith("Z:"):
        return "postal"
    if key.startswith("T:"):
        return "token"
    return "other"


def main(sample_size: int = 1000):
    print("=" * 80)
    print("BLOCKING DIAGNOSTIC — RENOVATED PIPELINE")
    print("=" * 80)

    s1_df, gt_map = load_sample_and_ground_truth(sample_size)
    country_indexes, target_locations = build_indexes_and_target_locations(gt_map)

    total_true = 0
    raw_hits = 0
    union_hits = 0
    pool_loss_hits = 0
    retained_bucket_miss_hits = 0
    country_miss_edges = 0

    family_edge_hits = Counter()
    family_entity_hits = Counter()
    likely_pruned_edges = 0
    current_candidate_counts = []

    missed_examples = []

    for _, row in s1_df.iterrows():
        s1_id = str(row["entity_id"])
        true_targets = gt_map.get(s1_id, set())

        if not true_targets:
            continue

        total_true += len(true_targets)

        country_key = clean_country(str(row["country"]))
        s1_name = clean_name(row["business_name"])
        s1_addr = clean_address(row["business_address"])
        s1_postal = extract_postal_code(s1_addr)

        c_idx = country_indexes.get(country_key)

        if c_idx is None or not s1_name:
            country_miss_edges += len(true_targets)
            continue

        keys = c_idx.generate_keys(s1_name, s1_postal)

        # Retained union of all relevant buckets.
        retained_union = set()
        # Track key-family hits separately for each true target edge.
        target_family_hits = defaultdict(set)

        for key in keys:
            bucket = c_idx.index.get(key, [])
            if not bucket:
                continue

            family = key_family(key)
            retained_union.update(bucket)

            for target_id in true_targets:
                loc = target_locations.get(target_id)
                if loc and loc[0] == country_key and loc[1] in bucket:
                    target_family_hits[target_id].add(family)

        for families in target_family_hits.values():
            for family in families:
                family_entity_hits[family] += 1

        # Current real retrieval path.
        current_indices = c_idx.get_candidates(
            s1_name,
            s1_postal,
            pool_size=DEFAULT_POOL_SIZE,
        )
        current_set = set(current_indices)
        current_candidate_counts.append(len(current_set))

        target_indices = {
            target_id: target_locations[target_id][1]
            for target_id in true_targets
            if target_id in target_locations
            and target_locations[target_id][0] == country_key
        }

        for target_id in true_targets:
            target_idx = target_indices.get(target_id)

            if target_idx is None:
                country_miss_edges += 1
                continue

            target_retrieved = target_idx in current_set
            target_in_retained_union = target_idx in retained_union

            if target_retrieved:
                raw_hits += 1

            if target_in_retained_union:
                union_hits += 1

            if target_in_retained_union and not target_retrieved:
                pool_loss_hits += 1

            if not target_in_retained_union and not target_retrieved:
                retained_bucket_miss_hits += 1

            # Look for strong evidence that a capped token bucket may have
            # removed the target. Exact/pair/postal buckets are not pruned.
            for key in keys:
                if not key.startswith("T:"):
                    continue
                bucket = c_idx.index.get(key, [])
                if len(bucket) >= c_idx.max_bucket_size and target_idx not in bucket:
                    likely_pruned_edges += 1
                    break

            if target_retrieved:
                for fam in target_family_hits.get(target_id, set()):
                    family_edge_hits[fam] += 1

            elif len(missed_examples) < 15:
                missed_examples.append({
                    "source1": s1_id,
                    "target": target_id,
                    "name": s1_name,
                    "postal": s1_postal,
                    "key_families_with_target": sorted(target_family_hits.get(target_id, set())),
                    "in_retained_key_union": target_in_retained_union,
                    "retrieved_by_current_pool": target_retrieved,
                })

    print("\n" + "=" * 80)
    print("DIAGNOSTIC RESULTS")
    print("=" * 80)

    print(f"Total true link edges                  : {total_true:,}")
    print(
        f"Current raw blocking recall            : "
        f"{raw_hits / total_true:.4f}"
        if total_true else "Current raw blocking recall            : 0.0000"
    )
    print(
        f"Recall if ALL retained key buckets used: "
        f"{union_hits / total_true:.4f}"
        if total_true else "Recall if ALL retained key buckets used: 0.0000"
    )
    print(
        f"True edges lost only due to pool/order  : {pool_loss_hits:,}"
    )
    print(
        f"True edges absent from retained keys   : {retained_bucket_miss_hits:,}"
    )
    print(
        f"True edges with no country/index       : {country_miss_edges:,}"
    )
    print(
        f"Likely token-bucket pruning losses     : {likely_pruned_edges:,}"
    )

    print("\nTrue-link retrieval by key family (edge count):")
    for fam in ["exact", "pair", "postal", "token"]:
        print(f"  {fam:7s}: {family_edge_hits[fam]:,}")

    print("\nEntity-level presence of a true link in each key family:")
    for fam in ["exact", "pair", "postal", "token"]:
        print(f"  {fam:7s}: {family_entity_hits[fam]:,} entities")

    if current_candidate_counts:
        avg = sum(current_candidate_counts) / len(current_candidate_counts)
        print(f"\nAverage current candidate count           : {avg:.2f}")
        print(f"Maximum current candidate count          : {max(current_candidate_counts)}")

    if missed_examples:
        print("\nExample missed true links (first 15):")
        for ex in missed_examples:
            print(
                f"  S1={ex['source1']} -> TRUE={ex['target']} | "
                f"name='{ex['name']}' | postal='{ex['postal']}' | "
                f"families={ex['key_families_with_target']} | "
                f"retained_union={ex['in_retained_key_union']}"
            )

    print("\nInterpretation guide:")
    print("  - Large pool/order loss  => increase/improve candidate retrieval.")
    print("  - Large retained-key miss => blocking keys are too strict/incomplete.")
    print("  - Large pruning loss      => token buckets are being truncated too aggressively.")
    print("  - Large country miss      => country normalization / dataset coverage issue.")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnose blocking recall on the training set."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of Source 1 training rows to sample.",
    )
    args = parser.parse_args()

    main(sample_size=args.sample_size)
