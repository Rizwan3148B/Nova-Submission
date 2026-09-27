"""
High-Recall, Compact-Candidate Inverted Index Blocking Engine.
Partitioned strictly by Country to eliminate cross-country search space and false merges.
Optimized with parallel flat arrays and tiered key retrieval.

CHANGES vs the original module (see CHANGELOG.md at the package root):
  - get_candidates() no longer hardcodes "50" as its retrieval pool size.
    In the original code, pipeline.py called get_candidates(name, postal)
    with no max_candidates argument, so it silently fell back to the
    function's default of 50 -- while matching.py's final cap was a
    *different*, separately-configured constant (MAX_CANDIDATES_PER_S1=15).
    Nothing was wrong syntactically, but it meant every Source 1 record was
    scored against up to 50 candidates even though at most 15 could ever
    survive into candidate_pairs.tsv, and the two numbers could drift apart
    over time with no error to catch it. DEFAULT_POOL_SIZE is now the one
    place that number lives, and pipeline.py/evaluate.py both import it
    explicitly instead of relying on an unlabeled default.
"""

from collections import defaultdict
from typing import List, Tuple, Dict, Any
from .normalization import clean_name, clean_address, extract_postal_code, get_core_name_tokens

# How many raw candidate indices the inverted index will hand back per query,
# BEFORE scoring/thresholding narrows that down to MAX_CANDIDATES_PER_S1 in
# matching.py. Kept generous (a few times MAX_CANDIDATES_PER_S1) so blocking
# recall isn't the bottleneck, but explicit and importable rather than a
# hidden default.
DEFAULT_POOL_SIZE = 75


class CountryBlockingIndex:
    """
    Inverted index for a single country's candidate generation.
    Uses parallel flat arrays for minimum memory footprint and high cache locality.
    """

    def __init__(
        self,
        country: str,
        max_bucket_size: int = 1500,
        token_prefix_len: int = 4,
        max_block_tokens: int = 2
    ):
        self.country = country
        self.max_bucket_size = max_bucket_size
        self.token_prefix_len = token_prefix_len
        self.max_block_tokens = max_block_tokens

        # Flat parallel arrays (saves ~500MB over millions of tuple objects)
        self.entity_ids: List[str] = []
        self.names: List[str] = []
        self.addrs: List[str] = []
        self.postals: List[str] = []

        # Inverted index: key -> list of integer record indices
        self.index: Dict[str, List[int]] = defaultdict(list)

    @property
    def num_records(self) -> int:
        return len(self.entity_ids)

    def generate_keys(self, norm_name: str, postal_code: str = "") -> List[str]:
        """Generate high-discriminative blocking keys for an entity."""
        if not norm_name:
            return []

        keys = []
        # 1. Exact normalized name key (highest precision)
        keys.append("E:" + norm_name)

        core_tokens = get_core_name_tokens(norm_name, max_tokens=self.max_block_tokens)
        prefixes = [t[:self.token_prefix_len] for t in core_tokens if len(t) >= 3]

        # 2. Order-independent pair prefix key (high selectivity)
        if len(prefixes) >= 2:
            pair = sorted([prefixes[0], prefixes[1]])
            keys.append(f"P:{pair[0]}_{pair[1]}")

        # 3. Individual distinctive token prefix keys
        for p in prefixes:
            keys.append("T:" + p)

        # 4. Postal code key (anchored with first 2 letters of name for selectivity)
        if postal_code and len(norm_name) >= 2:
            keys.append(f"Z:{postal_code}_{norm_name[:2]}")

        return keys

    def add_records(self, entity_ids: List[str], names: List[str], addresses: List[str]):
        """Batch-insert records from a chunk into the index."""
        start_idx = len(self.entity_ids)
        n = len(entity_ids)

        for i in range(n):
            e_id = str(entity_ids[i])
            n_name = names[i]
            n_addr = addresses[i]
            p_code = extract_postal_code(n_addr)

            rec_idx = start_idx + i
            self.entity_ids.append(e_id)
            self.names.append(n_name)
            self.addrs.append(n_addr)
            self.postals.append(p_code)

            keys = self.generate_keys(n_name, p_code)
            for k in keys:
                self.index[k].append(rec_idx)

    def prune_large_buckets(self) -> int:
        """
        Truncate pathological buckets to max_bucket_size rather than deleting them.
        Prevents losing recall for common single-token entities while stopping runaway explosion.
        """
        capped_count = 0
        for k in self.index:
            if len(self.index[k]) > self.max_bucket_size:
                if k.startswith("T:"):
                    self.index[k] = self.index[k][:self.max_bucket_size]
                    capped_count += 1
        return capped_count

    def get_candidates(self, norm_name: str, postal_code: str = "", pool_size: int = DEFAULT_POOL_SIZE) -> List[int]:
        """
        Retrieve candidate record indices using selective blocking keys.

        IMPORTANT: candidate ordering is deliberate. The previous implementation
        accumulated candidates in a set and then sliced the set to pool_size.
        Set iteration order is not a reliable notion of blocking-key priority, so
        a true match could be pushed out by an unrelated candidate even when the
        true record was present in an exact/pair/postal bucket.

        New strategy:
          1. Gather all non-empty generated keys.
          2. Rank the keys by bucket size (smallest / most selective first).
          3. Pull candidates from those buckets in that order, preserving bucket
             order and deduplicating without losing priority.
          4. Stop at pool_size.

        This does not change which keys exist or the pool size. It only makes
        the fixed-size candidate pool use the available evidence intelligently.
        """
        keys = self.generate_keys(norm_name, postal_code)
        if not keys or pool_size <= 0:
            return []

        # Resolve non-empty buckets once. More selective (smaller) buckets are
        # more valuable when the query has a hard pool-size budget.
        buckets = []
        seen_bucket_keys = set()

        for key in keys:
            if key in seen_bucket_keys:
                continue
            seen_bucket_keys.add(key)
            bucket = self.index.get(key)
            if bucket:
                buckets.append((len(bucket), key, bucket))

        if not buckets:
            return []

        # Smallest bucket first: exact names usually win, then selective pairs,
        # then postal/token buckets. The key text is a deterministic tie-break.
        buckets.sort(key=lambda x: (x[0], x[1]))

        candidates = []
        seen_records = set()

        for _, _, bucket in buckets:
            for rec_idx in bucket:
                if rec_idx in seen_records:
                    continue

                candidates.append(rec_idx)
                seen_records.add(rec_idx)

                if len(candidates) >= pool_size:
                    return candidates

        return candidates
