"""
Fast, robust text and entity normalization for Business Entity Resolution.
Supports multilingual records across US, India, and France.

CHANGES vs the original module (see CHANGELOG.md at the package root):
  - Added clean_country(): every place that partitions records by country
    (blocking.py, pipeline.py, evaluate.py) now goes through this function.
    Previously the raw country string was used as the dict key in one place
    and re-cast with str() in another; a stray space or casing difference
    ("India " vs "india") would silently create a second, empty index and
    turn real matches into false singletons. This is the single highest
    -leverage correctness fix in this renovation because it costs nothing
    and can only ever recover recall, never hurt it.
  - Added count_informative_tokens(): used by matching.py's new
    "generic name guard" to stop very short/legal-suffix-only names from
    auto-qualifying for a same-country, same-postal false merge.
"""

import re
import unicodedata

# Legal entity abbreviations and stopwords across US, India, France
LEGAL_AND_STOPWORDS = {
    # Common function words
    "the", "a", "an", "and", "of", "for", "in", "on", "at", "to", "by", "with",
    # French stopwords
    "de", "du", "des", "la", "le", "les", "et", "au", "aux", "sur",
    # US / General English legal forms
    "corp", "corporation", "corporate", "company", "co", "inc", "incorporated",
    "ltd", "limited", "pvt", "private", "llc", "llp", "pllc", "plc", "pc",
    "group", "grp", "holdings", "holding", "enterprises", "enterprise", "ent",
    "industries", "ind", "solutions", "sol", "services", "svc", "technologies",
    "technology", "tech", "international", "intl", "brothers", "bros",
    "associates", "assoc",
    # French legal forms
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "cie", "ets",
    "etablissements", "societe", "ste", "association",
    # Indian common enterprise suffixes
    "agency", "agencies", "traders", "stores", "mart", "works",
    # Web / domain noise
    "www", "com", "net", "org", "in", "fr"
}

# Standardized abbreviation dictionary
NAME_ABBREVIATIONS = {
    "corporation": "corp",
    "corporate": "corp",
    "company": "corp",
    "co": "corp",
    "compagnie": "corp",
    "cie": "corp",
    "societe": "ste",
    "ste": "ste",
    "etablissements": "ets",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "priv": "pvt",
    "&": "and",
    "enterprises": "ent",
    "enterprise": "ent",
    "industries": "ind",
    "international": "intl",
    "brothers": "bros",
    "associates": "assoc",
    "group": "grp",
    "solutions": "sol",
    "services": "svc",
    "technologies": "tech",
    "technology": "tech",
    "holding": "holdings",
}

ADDRESS_ABBREVIATIONS = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "circle": "cir",
    "court": "ct",
    "place": "pl",
    "square": "sq",
    "highway": "hwy",
    "parkway": "pkwy",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
    "suite": "ste",
    # French address terms
    "bd": "blvd",
    "av": "ave",
    "rue": "st",
    "all": "allee",
    "allee": "allee",
    "rte": "route",
    "chemin": "ch",
    "impasse": "imp",
    "cours": "crs",
    # Directional
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    # Landmark noise reduction
    "near": "",
    "opp": "",
    "opposite": "",
    "beside": "",
    "behind": "",
}

PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
POSTAL_CODE_RE = re.compile(r"\b\d{5,6}\b")


def to_ascii(text: str) -> str:
    """Normalize unicode text into ASCII with fast-path for pure ASCII strings."""
    if not text:
        return ""
    text_str = str(text)
    if text_str.isascii():
        return text_str
    return unicodedata.normalize("NFKD", text_str).encode("ASCII", "ignore").decode("utf-8")


def clean_name(raw_name: str) -> str:
    """
    Standardize a business name:
    NFKD ASCII normalization, lowercasing, punctuation stripping, and abbreviation mapping.
    """
    if not raw_name or not isinstance(raw_name, str):
        return ""
    text = to_ascii(raw_name).lower()
    text = PUNCT_RE.sub(" ", text)
    tokens = [NAME_ABBREVIATIONS.get(t, t) for t in text.split()]
    return " ".join(tokens)


def clean_address(raw_address: str) -> str:
    """
    Standardize a business address:
    NFKD ASCII normalization, abbreviation mapping, and whitespace normalization.
    """
    if not raw_address or not isinstance(raw_address, str):
        return ""
    text = to_ascii(raw_address).lower()
    text = PUNCT_RE.sub(" ", text)
    tokens = []
    for t in text.split():
        rep = ADDRESS_ABBREVIATIONS.get(t, t)
        if rep:
            tokens.append(rep)
    return " ".join(tokens)


def clean_country(raw_country: str) -> str:
    """
    Standardize a country label so it can be used safely as a dict/index key.

    NEW in this renovation. The pipeline partitions ALL blocking and matching
    work by country, so this function is the single most important piece of
    plumbing in the whole system: every place that reads a country value
    (building indexes from Source 2/3, or looking one up for a Source 1
    record) must call this, or a whitespace/case mismatch between files will
    silently drop that record into an empty bucket and it will come out as a
    false singleton with zero code-level errors to warn you.

    Example:
        "India"  -> "india"
        " India" -> "india"
        "INDIA"  -> "india"
        "France" -> "france"
    """
    if not raw_country or not isinstance(raw_country, str):
        return ""
    return raw_country.strip().lower()


def extract_postal_code(address: str) -> str:
    """Extract a 5 or 6 digit postal code if present in the address."""
    if not address:
        return ""
    match = POSTAL_CODE_RE.search(address)
    return match.group(0) if match else ""


def get_core_name_tokens(norm_name: str, max_tokens: int = 3):
    """
    Extract distinctive enterprise name tokens, filtering out legal suffixes and stopwords.
    Returns list of tokens sorted by length (descending).
    """
    if not norm_name:
        return []
    tokens = norm_name.split()
    informative = [t for t in tokens if t not in LEGAL_AND_STOPWORDS and len(t) >= 3 and not t.isdigit()]
    if not informative:
        informative = [t for t in tokens if len(t) >= 2 and not t.isdigit()]
    if not informative:
        informative = tokens

    # Deduplicate preserving order
    unique_tokens = list(dict.fromkeys(informative))
    # Sort longest first
    unique_tokens.sort(key=lambda t: (len(t), t), reverse=True)
    return unique_tokens[:max_tokens]


def count_informative_tokens(norm_name: str) -> int:
    """
    NEW in this renovation. Counts tokens in a normalized name that are NOT
    legal suffixes / stopwords and are at least 2 characters long.

    Used by matching.py's "generic name guard": a name like "sri traders" or
    just "corp pvt ltd" (after suffix stripping, effectively empty) is too
    generic for a same-name-different-real-business collision to be ruled
    out by name similarity alone. This does not decide anything by itself
    -- it just tells the matcher when it needs a stronger address/postal
    signal before granting a high score.
    """
    if not norm_name:
        return 0
    tokens = norm_name.split()
    return sum(1 for t in tokens if t not in LEGAL_AND_STOPWORDS and len(t) >= 2 and not t.isdigit())
