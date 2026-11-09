import pytest

from spam_shield import config
from spam_shield.predict import (
    InvalidMessageError,
    Prediction,
    predict_message,
    risk_level,
    validate_message,
)


def test_obvious_spam_is_high_risk():
    prediction = predict_message("WIN a FREE iPhone!!! Reply YES to claim your prize now")
    assert prediction.label == "spam"
    assert prediction.risk_level == "high"


def test_normal_message_is_safe():
    prediction = predict_message("Hey, are we still meeting for lunch tomorrow at 1?")
    assert prediction.label == "ham"
    assert prediction.risk_level == "safe"


def test_probabilities_cover_all_labels_and_sum_to_one():
    prediction = predict_message("Your account has been suspended. Verify your password now")
    assert isinstance(prediction, Prediction)
    assert list(prediction.probabilities) == config.LABELS
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert prediction.risk_score == pytest.approx(1 - prediction.probabilities["ham"])


@pytest.mark.parametrize("bad_input", ["", "   \n\t ", None, 42])
def test_invalid_input_is_rejected(bad_input):
    with pytest.raises(InvalidMessageError):
        validate_message(bad_input)


def test_too_long_message_is_rejected():
    with pytest.raises(InvalidMessageError, match="too long"):
        validate_message("a" * (config.MAX_MESSAGE_CHARS + 1))


def test_input_is_stripped():
    assert validate_message("  hello  ") == "hello"


@pytest.mark.parametrize(
    ("risk", "expected"),
    [(0.0, "safe"), (0.29, "safe"), (0.3, "suspicious"), (0.69, "suspicious"),
     (0.7, "high"), (1.0, "high")],
)
def test_risk_levels(risk, expected):
    assert risk_level(risk) == expected
