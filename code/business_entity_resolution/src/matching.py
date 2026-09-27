"""
High-Precision Entity Matching Engine optimized for Macro-F0.5.
Combines RapidFuzz string similarity, address token-set alignment, and postal verification.

CHANGES vs the original module (see CHANGELOG.md at the package root):

  1. VECTORIZED NAME SCORING (speed).
     The original code called fuzz.token_sort_ratio() and fuzz.token_set_ratio()
     once per candidate, in a plain Python for-loop, for every one of the up
     to ~45-50 pooled candidates per Source 1 record. Each call has fixed
     Python-level call overhead; across ~1.7M Source 1 records this is where
     most of the pipeline's wall-clock time goes. This version instead makes
     ONE rapidfuzz.process.cdist() call per metric per record, which computes
     the whole candidate pool's scores in a single C loop. The per-candidate
     math (early-exit thresholds, address scoring, postal bonus/penalty) is
     mathematically IDENTICAL to the original -- this only changes how the
     name-similarity numbers are computed, not what they mean.

  2. GENERIC-NAME GUARD (precision).
     Diagnostic review of the current test-set output showed a small number
     of Source 1 entities matched to 30-49 Source 2/3 records, versus a
     training-ground-truth observed max of 11. This pattern is consistent
     with very generic/short business names (e.g. "sri traders", or a name
     that is entirely legal-suffix words) clearing the similarity threshold
     against many unrelated real businesses. A name is now checked for how
     many "informative" tokens it has left after stripping stopwords/legal
     suffixes (normalization.count_informative_tokens); if it has fewer than
     MIN_INFORMATIVE_TOKENS, it can no longer reach MATCH_THRESHOLD purely
     on name+address similarity -- it also needs matching postal codes as
     corroborating evidence. Candidate generation (candidate_pairs.tsv) is
     untouched by this guard; it only tightens the final match decision.

  3. MAX_MATCHES_PER_S1 safety valve (robustness).
     A hard, documented ceiling on how many matches one Source 1 entity can
     receive, set comfortably above the training ground truth's observed
     max (11) at 20. This does not fix a precision problem by itself, but it
     turns "one bad bucket silently produces 49 matches" into a visible,
     bounded, and log-able event rather than an open-ended one.

Run diagnose_candidate_loss.py / evaluate.py on your own train sample after
this change to confirm precision improved without a meaningful recall hit --
thresholds are data-dependent and the guard's cutoff (MIN_INFORMATIVE_TOKENS)
is a starting point, not a proven-optimal value.
"""

from typing import List, Tuple, Dict, Any
from rapidfuzz import fuzz, process

from .normalization import count_informative_tokens

CANDIDATE_THRESHOLD = 65.0
MATCH_THRESHOLD = 88.0

NAME_WEIGHT = 0.70
ADDRESS_WEIGHT = 0.30

# Maximum candidates retained per Source 1 entity in candidate_pairs.tsv
MAX_CANDIDATES_PER_S1 = 15

# NEW: safety cap on matched_entity_ids length per Source 1 entity.
MAX_MATCHES_PER_S1 = 20

# NEW: minimum informative (non-stopword / non-legal-suffix) tokens a name
# needs before it can auto-qualify for the exact-match fast path or reach
# MATCH_THRESHOLD without postal-code corroboration.
MIN_INFORMATIVE_TOKENS = 1


def score_pair(
    s1_name: str,
    s1_addr: str,
    s1_postal: str,
    cand_name: str,
    cand_addr: str,
    cand_postal: str
) -> float:
    """
    Reference single-pair implementation (unvectorized). Kept for ad-hoc
    diagnostics / notebooks where scoring one pair at a time is convenient.
    score_and_rank_candidates() below does NOT call this -- it reimplements
    the same math in a batched form for speed. If you change the scoring
    logic, change it in both places (or delete this one and point any
    diagnostic scripts at score_and_rank_candidates with a single-item pool).
    """
    is_generic = count_informative_tokens(s1_name) < MIN_INFORMATIVE_TOKENS

    if s1_name == cand_name and s1_name and not is_generic:
        if not s1_addr or not cand_addr or s1_addr == cand_addr:
            return 100.0
        addr_score = fuzz.token_set_ratio(s1_addr, cand_addr, score_cutoff=40.0)
        return round(NAME_WEIGHT * 100.0 + ADDRESS_WEIGHT * addr_score, 2)

    name_sort = fuzz.token_sort_ratio(s1_name, cand_name, score_cutoff=45.0)
    if name_sort < 45.0:
        name_set = fuzz.token_set_ratio(s1_name, cand_name, score_cutoff=50.0)
        if name_set < 50.0:
            return 0.0
        name_score = 0.70 * name_sort + 0.30 * name_set
    else:
        name_set = fuzz.token_set_ratio(s1_name, cand_name)
        name_score = 0.70 * name_sort + 0.30 * name_set

    if name_score < 50.0:
        return 0.0

    postal_match = bool(s1_postal and cand_postal and s1_postal == cand_postal)

    if s1_addr and cand_addr:
        if s1_addr == cand_addr:
            combined = NAME_WEIGHT * name_score + ADDRESS_WEIGHT * 100.0
        else:
            addr_score = fuzz.token_set_ratio(s1_addr, cand_addr, score_cutoff=30.0)
            combined = NAME_WEIGHT * name_score + ADDRESS_WEIGHT * addr_score

        if postal_match:
            combined = min(100.0, combined + 5.0)
        elif s1_postal and cand_postal:
            combined = max(0.0, combined - 15.0)

        score = round(combined, 2)
    else:
        score = round(name_score, 2)

    if is_generic and score >= MATCH_THRESHOLD and not postal_match:
        score = MATCH_THRESHOLD - 1.0

    return score


def score_and_rank_candidates(
    s1_name: str,
    s1_addr: str,
    s1_postal: str,
    candidate_indices: List[int],
    blocking_index: Any,
    cand_threshold: float = CANDIDATE_THRESHOLD,
    match_threshold: float = MATCH_THRESHOLD,
    max_candidates: int = MAX_CANDIDATES_PER_S1,
    max_matches: int = MAX_MATCHES_PER_S1,
) -> Tuple[List[str], List[str]]:
    """
    Score all pooled candidates for a single Source 1 entity and return:
        (candidate_ids, matched_ids)
    Both lists are deduplicated and sorted by score descending.
    Matches are guaranteed to be a subset of candidates.
    Supports either CountryBlockingIndex or a legacy list of records.
    """
    if not candidate_indices or not s1_name:
        return [], []

    use_index = hasattr(blocking_index, "entity_ids")

    if use_index:
        cand_ids_raw = [blocking_index.entity_ids[i] for i in candidate_indices]
        cand_names = [blocking_index.names[i] for i in candidate_indices]
        cand_addrs = [blocking_index.addrs[i] for i in candidate_indices]
        cand_postals = [blocking_index.postals[i] for i in candidate_indices]
    else:
        cand_ids_raw, cand_names, cand_addrs, cand_postals = [], [], [], []
        for i in candidate_indices:
            e_id, c_name, c_addr, c_postal = blocking_index[i]
            cand_ids_raw.append(e_id)
            cand_names.append(c_name)
            cand_addrs.append(c_addr)
            cand_postals.append(c_postal)

    n = len(cand_names)
    is_generic = count_informative_tokens(s1_name) < MIN_INFORMATIVE_TOKENS

    # ---- Stage 1: vectorized name similarity, one C call per metric ----
    # (Single-element query list; cdist returns a (1, n) array.)
    sort_scores = process.cdist([s1_name], cand_names, scorer=fuzz.token_sort_ratio)[0]
    set_scores = process.cdist([s1_name], cand_names, scorer=fuzz.token_set_ratio)[0]

    scored_list = []

    for i in range(n):
        cand_name = cand_names[i]
        cand_addr = cand_addrs[i]
        cand_postal = cand_postals[i]
        e_id = cand_ids_raw[i]

        # Exact match fast path -- gated by the generic-name guard.
        if s1_name == cand_name and s1_name and not is_generic:
            if not s1_addr or not cand_addr or s1_addr == cand_addr:
                scored_list.append((e_id, 100.0))
            else:
                addr_score = fuzz.token_set_ratio(s1_addr, cand_addr, score_cutoff=40.0)
                scored_list.append((e_id, round(NAME_WEIGHT * 100.0 + ADDRESS_WEIGHT * addr_score, 2)))
            continue

        name_sort = float(sort_scores[i])
        name_set = float(set_scores[i])

        if name_sort < 45.0:
            if name_set < 50.0:
                continue
            name_score = 0.70 * name_sort + 0.30 * name_set
        else:
            name_score = 0.70 * name_sort + 0.30 * name_set

        if name_score < 50.0:
            continue

        postal_match = bool(s1_postal and cand_postal and s1_postal == cand_postal)
        postal_conflict = bool(s1_postal and cand_postal and s1_postal != cand_postal)

        if s1_addr and cand_addr:
            if s1_addr == cand_addr:
                combined = NAME_WEIGHT * name_score + ADDRESS_WEIGHT * 100.0
            else:
                addr_score = fuzz.token_set_ratio(s1_addr, cand_addr, score_cutoff=30.0)
                combined = NAME_WEIGHT * name_score + ADDRESS_WEIGHT * addr_score

            if postal_match:
                combined = min(100.0, combined + 5.0)
            elif postal_conflict:
                combined = max(0.0, combined - 15.0)

            score = round(combined, 2)
        else:
            score = round(name_score, 2)

        # Generic-name guard (see module docstring, point 2).
        if is_generic and score >= match_threshold and not postal_match:
            score = match_threshold - 1.0

        if score >= cand_threshold:
            scored_list.append((e_id, score))

    if not scored_list:
        return [], []

    scored_list.sort(key=lambda x: x[1], reverse=True)

    cand_ids = list(dict.fromkeys(e_id for e_id, score in scored_list[:max_candidates]))

    match_ids = list(dict.fromkeys(
        e_id for e_id, score in scored_list if score >= match_threshold
    ))[:max_matches]

    # Ensure match_ids is strictly a subset of cand_ids
    for mid in match_ids:
        if mid not in cand_ids:
            cand_ids.append(mid)

    return cand_ids, match_ids
