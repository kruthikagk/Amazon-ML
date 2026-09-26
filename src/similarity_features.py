"""
Reusable similarity features for business entity resolution.

Designed for candidate pairs between Source 1 and Source 2/3.

Input:
    Two entity records containing:
        entity_id
        business_name_clean
        business_address_clean
        country

Output:
    Dictionary of pairwise similarity features.
"""

from typing import Any, Dict, Iterable, Set

from rapidfuzz import fuzz


def safe_string(value: Any) -> str:
    """Convert a value to a safe lowercase string."""
    if value is None:
        return ""

    if isinstance(value, float):
        # Handles NaN without requiring pandas.
        if value != value:
            return ""

    return str(value).strip().lower()


def is_missing(value: Any) -> int:
    """Return 1 if a field is missing/empty, otherwise 0."""
    return int(safe_string(value) == "")


def token_set(value: Any) -> Set[str]:
    """Return whitespace-separated tokens as a set."""
    text = safe_string(value)

    if not text:
        return set()

    return set(text.split())


def exact_match(left: Any, right: Any) -> int:
    """Binary exact-match feature."""
    left_text = safe_string(left)
    right_text = safe_string(right)

    if not left_text or not right_text:
        return 0

    return int(left_text == right_text)


def fuzzy_similarity(left: Any, right: Any) -> float:
    """
    Normalized fuzzy similarity using RapidFuzz ratio.

    Returns a value between 0 and 1.
    """
    left_text = safe_string(left)
    right_text = safe_string(right)

    if not left_text or not right_text:
        return 0.0

    return fuzz.ratio(left_text, right_text) / 100.0


def jaccard_similarity(left: Any, right: Any) -> float:
    """
    Token-set Jaccard similarity.

    J(A,B) = |A intersection B| / |A union B|
    """
    left_tokens = token_set(left)
    right_tokens = token_set(right)

    if not left_tokens or not right_tokens:
        return 0.0

    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens

    return len(intersection) / len(union)


def levenshtein_similarity(left: Any, right: Any) -> float:
    """
    Normalized Levenshtein similarity using RapidFuzz.

    Returns a value between 0 and 1.
    """
    left_text = safe_string(left)
    right_text = safe_string(right)

    if not left_text or not right_text:
        return 0.0

    return fuzz.ratio(left_text, right_text) / 100.0


def length_features(value: Any) -> Dict[str, int]:
    """Return character and token length features."""
    text = safe_string(value)

    return {
        "char_length": len(text),
        "token_count": len(text.split()) if text else 0,
    }


def length_pair_features(left: Any, right: Any) -> Dict[str, float]:
    """Return length and length-difference features for a pair."""
    left_text = safe_string(left)
    right_text = safe_string(right)

    left_length = len(left_text)
    right_length = len(right_text)

    left_tokens = len(left_text.split()) if left_text else 0
    right_tokens = len(right_text.split()) if right_text else 0

    max_char_length = max(left_length, right_length)
    max_token_count = max(left_tokens, right_tokens)

    return {
        "left_char_length": left_length,
        "right_char_length": right_length,
        "char_length_diff": abs(left_length - right_length),
        "char_length_ratio": (
            min(left_length, right_length) / max_char_length
            if max_char_length > 0
            else 0.0
        ),
        "left_token_count": left_tokens,
        "right_token_count": right_tokens,
        "token_count_diff": abs(left_tokens - right_tokens),
        "token_count_ratio": (
            min(left_tokens, right_tokens) / max_token_count
            if max_token_count > 0
            else 0.0
        ),
    }


def string_features(left: Any, right: Any, prefix: str) -> Dict[str, float]:
    """
    Generate all string-comparison features for one field.

    prefix:
        "name" or "address"
    """
    features = {
        f"{prefix}_exact": exact_match(left, right),
        f"{prefix}_similarity": fuzzy_similarity(left, right),
        f"{prefix}_jaccard": jaccard_similarity(left, right),
        f"{prefix}_levenshtein": levenshtein_similarity(left, right),
        f"{prefix}_left_missing": is_missing(left),
        f"{prefix}_right_missing": is_missing(right),
    }

    lengths = length_pair_features(left, right)

    for key, value in lengths.items():
        features[f"{prefix}_{key}"] = value

    return features


def compute_pair_features(
    left_record: Dict[str, Any],
    right_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute reusable similarity features for one candidate pair.

    Expected fields:
        entity_id
        business_name_clean
        business_address_clean
        country
    """

    left_name = left_record.get("business_name_clean", "")
    right_name = right_record.get("business_name_clean", "")

    left_address = left_record.get("business_address_clean", "")
    right_address = right_record.get("business_address_clean", "")

    left_country = left_record.get("country", "")
    right_country = right_record.get("country", "")

    features = {
        "source1_entity_id": left_record.get("entity_id"),
        "candidate_entity_id": right_record.get("entity_id"),

        "country_match": exact_match(left_country, right_country),

        "left_country_missing": is_missing(left_country),
        "right_country_missing": is_missing(right_country),
    }

    features.update(
        string_features(
            left_name,
            right_name,
            "name",
        )
    )

    features.update(
        string_features(
            left_address,
            right_address,
            "address",
        )
    )

    return features


def compute_features_for_pairs(
    pairs: Iterable[tuple],
) -> list:
    """
    Compute features for multiple candidate pairs.

    Each item in pairs must be:

        (left_record, right_record)

    Returns:
        List of feature dictionaries.
    """

    return [
        compute_pair_features(left_record, right_record)
        for left_record, right_record in pairs
    ]