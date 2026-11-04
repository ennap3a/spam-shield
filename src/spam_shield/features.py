"""Metadata features: simple numeric facts about the RAW message text.

These are computed on the original text (before cleaning), because cleaning
removes exactly the clues we want to count: capitals, "!", "$", links.
"""

import pandas as pd

from spam_shield.preprocessing import EMAIL_PATTERN, URL_PATTERN


def extract_metadata(texts) -> pd.DataFrame:
    """Return one row per message with numeric features about its raw text."""
    texts = pd.Series(texts).fillna("").astype(str)
    char_count = texts.str.len()
    safe_length = char_count.clip(lower=1)   # avoid dividing by 0 for empty messages

    return pd.DataFrame({
        "char_count": char_count,
        "word_count": texts.str.split().str.len(),
        "url_count": texts.str.count(URL_PATTERN.pattern),
        "email_count": texts.str.count(EMAIL_PATTERN.pattern),
        "exclamation_count": texts.str.count(r"!"),
        "currency_count": texts.str.count(r"[$£€]"),
        "uppercase_ratio": texts.str.count(r"[A-Z]") / safe_length,
        "digit_ratio": texts.str.count(r"\d") / safe_length,
    })