"""Service layer: the ONE place the app (or anything else) uses to classify a message.

The app never touches the model, spaCy or pandas directly. It calls
predict_message(text) and gets back a simple Prediction. If we change the model
later, only this file needs to know.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from spam_shield import config
from spam_shield.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

RISK_MESSAGES = {
    "safe": "Looks like a normal message.",
    "suspicious": "Be careful: this message has some warning signs.",
    "high": "High risk: this message is very likely harmful.",
}


class InvalidMessageError(ValueError):
    """The input cannot be classified (not text, empty, or too long)."""


@dataclass(frozen=True)
class Prediction:
    """Everything the app needs to show for one message."""

    label: str                      # "ham", "spam", "phishing" or "toxic"
    probabilities: dict[str, float] # one probability per label, they sum to 1
    risk_score: float               # 1 - P(ham): how likely it is NOT a normal message
    risk_level: str                 # "safe", "suspicious" or "high"

    @property
    def risk_message(self) -> str:
        return RISK_MESSAGES[self.risk_level]


@lru_cache(maxsize=1)
def load_model(path: Path = config.MODEL_FILE):
    """Load the trained Pipeline once; later calls reuse the same copy."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run `python -m spam_shield.train` first."
        )
    logger.info("Loading model from %s", path)
    return joblib.load(path)


def validate_message(text) -> str:
    """Return the cleaned-up input, or raise InvalidMessageError with a clear reason."""
    if not isinstance(text, str):
        raise InvalidMessageError("The message must be text.")
    text = text.strip()
    if not text:
        raise InvalidMessageError("The message is empty.")
    if len(text) > config.MAX_MESSAGE_CHARS:
        raise InvalidMessageError(
            f"The message is too long ({len(text):,} characters). "
            f"The limit is {config.MAX_MESSAGE_CHARS:,}."
        )
    return text


def risk_level(risk_score: float) -> str:
    """Turn a risk score (0..1) into the level shown in the app."""
    if risk_score >= config.HIGH_RISK:
        return "high"
    if risk_score >= config.SUSPICIOUS_RISK:
        return "suspicious"
    return "safe"


def predict_message(text) -> Prediction:
    """Classify one raw message exactly the way the training data was processed."""
    text = validate_message(text)
    model = load_model()

    # The model expects the same two columns it was trained on.
    row = pd.DataFrame({"text_content": [text], "processed_text": [preprocess_text(text)]})
    scores = model.predict_proba(row)[0]
    by_label = dict(zip(model.classes_, scores))
    probabilities = {label: float(by_label[label]) for label in config.LABELS}

    label = max(probabilities, key=probabilities.get)
    risk = 1.0 - probabilities["ham"]
    prediction = Prediction(label, probabilities, risk, risk_level(risk))

    logger.info("Predicted %s (risk %.2f, %d chars)", label, risk, len(text))
    return prediction
