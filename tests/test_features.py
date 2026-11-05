import pytest
import pandas as pd
from spam_shield.features import build_feature_transformer, extract_metadata


def test_metadata_on_spam_like_message():
    features = extract_metadata(["WIN $500 NOW!!! Visit http://x.co"]).iloc[0]
    assert features["char_count"] == 33
    assert features["word_count"] == 5
    assert features["url_count"] == 1
    assert features["exclamation_count"] == 3
    assert features["currency_count"] == 1
    assert features["uppercase_ratio"] == pytest.approx(7 / 33)
    assert features["digit_ratio"] == pytest.approx(3 / 33)


def test_metadata_on_empty_message_is_all_zeros():
    features = extract_metadata([""]).iloc[0]
    assert (features == 0).all()


def make_fake_messages() -> pd.DataFrame:
    return pd.DataFrame({
        "text_content": ["WIN a FREE prize!!!", "hello friend", "see you tomorrow",
                         "FREE money $$$ now", "hi there", "claim your prize"],
        "processed_text": ["win free prize", "hello friend", "tomorrow",
                           "free money", "hi", "claim prize"],
    })


def test_feature_transformer_shape():
    transformer = build_feature_transformer(min_df=1)
    X = transformer.fit_transform(make_fake_messages())
    vocabulary_size = len(transformer.named_transformers_["tfidf"].vocabulary_)
    assert X.shape == (6, vocabulary_size + 7)    # word columns + 7 metadata columns


def test_new_message_gets_same_columns():
    transformer = build_feature_transformer(min_df=1)
    X_train = transformer.fit_transform(make_fake_messages())
    new = pd.DataFrame({"text_content": ["totally new words"], "processed_text": ["totally new word"]})
    X_new = transformer.transform(new)            # transform only, NO fit!
    assert X_new.shape[1] == X_train.shape[1]


def test_all_features_are_not_negative():
    X = build_feature_transformer(min_df=1).fit_transform(make_fake_messages())
    assert X.min() >= 0                           # Naive Bayes needs this    