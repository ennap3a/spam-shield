"""Prepare model-ready data: preprocess every message once, then split train/test.

spaCy takes a few minutes on the full dataset, so we run it once here and
save the result, instead of repeating it every time we train a model.
"""

import time

import pandas as pd
from sklearn.model_selection import train_test_split

from spam_shield import config
from spam_shield.preprocessing import preprocess_texts


def add_processed_text(df: pd.DataFrame) -> pd.DataFrame:
    """Add a processed_text column: clean -> lemmas -> no stop words."""
    df = df.copy()
    df["processed_text"] = preprocess_texts(df["text_content"].tolist())
    return df


def split_train_test(df: pd.DataFrame, test_size: float, random_state: int):
    """Split so every (label, source) group keeps the same share in train and test."""
    strata = df["label"] + "_" + df["source"]
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=strata,
        random_state=random_state,
    )
    return train_df, test_df

def load_split(path) -> pd.DataFrame:
    """Read a train/test CSV; empty processed texts come back as "" (not NaN) """
    df = pd.read_csv(path)
    df["processed_text"] = df["processed_text"].fillna("")
    return df   

if __name__ == "__main__":
    df = pd.read_csv(config.PROCESSED_DATASET_FILE)

    start = time.time()
    df = add_processed_text(df)
    print(f"Preprocessed {len(df):,} messages in {time.time() - start:.0f} seconds")

    train_df, test_df = split_train_test(df, config.TEST_SIZE, config.RANDOM_STATE)
    train_df.to_csv(config.TRAIN_FILE, index=False)
    test_df.to_csv(config.TEST_FILE, index=False)

    print(f"Train: {len(train_df):,} rows  ->  {config.TRAIN_FILE.name}")
    print(f"Test:  {len(test_df):,} rows  ->  {config.TEST_FILE.name}")
    print("\nShare of each group (train vs test):")
    print(pd.DataFrame({
        "train": train_df.groupby(["label", "source"]).size() / len(train_df),
        "test": test_df.groupby(["label", "source"]).size() / len(test_df),
    }).round(3))