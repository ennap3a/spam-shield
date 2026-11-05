"""Metadata features: simple numeric facts about the RAW message text.

These are computed on the original text (before cleaning), because cleaning
removes exactly the clues we want to count: capitals, "!", "$", links.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

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

# uppercase_ratio is NOT used by the model: the email dataset was lowercased
# by its creators, so it reveals the source (email vs sms/chat), not the label.
MODEL_METADATA_COLUMNS = [
    "char_count",
    "word_count",
    "url_count",
    "email_count",
    "exclamation_count",
    "currency_count",
    "digit_ratio",
]


def metadata_for_model(texts) -> pd.DataFrame:
    """Metadata features, keeping only the columns the model is allowed to use."""
    return extract_metadata(texts)[MODEL_METADATA_COLUMNS]


def build_feature_transformer(min_df: int = 3) -> ColumnTransformer:
    """TF-IDF on processed_text + scaled metadata on raw text_content, side by side."""
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.9,
        max_features=50_000,
        sublinear_tf=True,
    )
    metadata = make_pipeline(
        FunctionTransformer(metadata_for_model),   # raw text -> 7 numbers
        FunctionTransformer(np.log1p),             # shrink huge values
        MinMaxScaler(),                            # squeeze into 0..1
    )
    return ColumnTransformer([
        ("tfidf", tfidf, "processed_text"),
        ("metadata", metadata, "text_content")
    ])
       