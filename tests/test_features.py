import pytest

from spam_shield.features import extract_metadata


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