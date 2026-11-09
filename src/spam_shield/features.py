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

# Columns the model is NOT allowed to use, because they reveal the source
# (email vs sms/chat) instead of the label:
# - uppercase_ratio: the email dataset was lowercased by its creators.
# - char_count, word_count: emails are long, SMS/chat are short. Task 5.1 showed
#   that removing them keeps validation macro-F1 almost the same but improves
#   results on the hand-written challenge set.
MODEL_METADATA_COLUMNS = [
    "url_count",
    "email_count",
    "exclamation_count",
    "currency_count",
    "digit_ratio",
]


def metadata_for_model(texts, columns=MODEL_METADATA_COLUMNS) -> pd.DataFrame:
    """Metadata features, keeping only the columns the model is allowed to use."""
    return extract_metadata(texts)[list(columns)]


def build_feature_transformer(
    min_df: int = 3, metadata_columns=MODEL_METADATA_COLUMNS
) -> ColumnTransformer:
    """TF-IDF on processed_text + scaled metadata on raw text_content, side by side."""
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.9,
        max_features=50_000,
        sublinear_tf=True,
    )
    metadata = make_pipeline(
        FunctionTransformer(                       # raw text -> a few numbers
            metadata_for_model, kw_args={"columns": list(metadata_columns)}
        ),
        FunctionTransformer(np.log1p),             # shrink huge values
        MinMaxScaler(),                            # squeeze into 0..1
    )
    parts = [("tfidf", tfidf, "processed_text")]
    if metadata_columns:                           # an empty list means "words only"
        parts.append(("metadata", metadata, "text_content"))
    return ColumnTransformer(parts)
       