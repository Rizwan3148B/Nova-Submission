#!/usr/bin/env python3
"""
Main execution entrypoint for Business Entity Resolution.
Run from repository root:
    python code/business_entity_resolution/run.py
Or from the directory:
    python run.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from src.pipeline import run_pipeline
from src.matching import CANDIDATE_THRESHOLD, MATCH_THRESHOLD, MAX_CANDIDATES_PER_S1, MAX_MATCHES_PER_S1
from src.blocking import DEFAULT_POOL_SIZE


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument(
        "--data-dir", type=str,
        default=str(curr_dir.parent.parent / "dataset" / "test"),
        help="Path to dataset directory containing test_source1/2/3.tsv"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(curr_dir.parent.parent / "output"),
        help="Path to output directory for matching_results.tsv and candidate_pairs.tsv"
    )
    parser.add_argument("--cand-threshold", type=float, default=CANDIDATE_THRESHOLD)
    parser.add_argument("--match-threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES_PER_S1)
    parser.add_argument(
        "--max-matches", type=int, default=MAX_MATCHES_PER_S1,
        help="Safety cap on matched_entity_ids length per Source 1 entity."
    )
    parser.add_argument(
        "--pool-size", type=int, default=DEFAULT_POOL_SIZE,
        help="How many raw candidates blocking retrieves per query before scoring."
    )

    args = parser.parse_args()

    run_pipeline(
        data_dir=Path(args.data_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        cand_threshold=args.cand_threshold,
        match_threshold=args.match_threshold,
        max_candidates=args.max_candidates,
        max_matches=args.max_matches,
        pool_size=args.pool_size,
    )


if __name__ == "__main__":
    main()
