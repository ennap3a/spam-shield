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
import numpy as np
import pandas as pd

from spam_shield import config
from spam_shield.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

RISK_MESSAGES = {
    "safe": "This looks like a normal message.",
    "suspicious": "Be careful: this message has some warning signs.",
    "high": "This message is very likely harmful.",
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


def _to_model_input(text: str) -> pd.DataFrame:
    """The model expects the same two columns it was trained on."""
    return pd.DataFrame({"text_content": [text], "processed_text": [preprocess_text(text)]})


def word_weights(model) -> np.ndarray:
    """How much each word pushes towards each label: one row per label.

    The calibrated model holds 5 LinearSVCs (one per calibration fold); we
    average their coefficients. A positive weight pushes TOWARDS that label.
    """
    calibrated = model.named_steps["classifier"].calibrated_classifiers_
    return np.mean([c.estimator.coef_ for c in calibrated], axis=0)


def vocabulary(model) -> np.ndarray:
    """The words/word pairs of the TF-IDF part, in column order."""
    return model.named_steps["features"].named_transformers_["tfidf"].get_feature_names_out()


def explain_prediction(text, label: str, top_n: int = 5) -> list[tuple[str, float]]:
    """The words in THIS message that pushed most towards `label`.

    contribution = (TF-IDF value of the word in this message) x (word weight)
    Only words with a positive contribution are returned, biggest first.
    """
    text = validate_message(text)
    model = load_model()
    X = model.named_steps["features"].transform(_to_model_input(text)).tocsr()
    weights = word_weights(model)[list(model.classes_).index(label)]
    words = vocabulary(model)

    contributions = []
    for column, value in zip(X.indices, X.data):
        if column < len(words):                   # TF-IDF columns come first
            contribution = value * weights[column]
            if contribution > 0:
                contributions.append((words[column], float(contribution)))
    return sorted(contributions, key=lambda pair: pair[1], reverse=True)[:top_n]


def top_words_per_label(model, top_n: int = 15) -> dict[str, list[str]]:
    """The words the model relies on most for each label (used by the dashboard)."""
    words = vocabulary(model)
    weights = word_weights(model)[:, : len(words)]   # TF-IDF columns only
    result = {}
    for row, label in enumerate(model.classes_):
        best = np.argsort(weights[row])[::-1][:top_n]
        result[str(label)] = [str(words[i]) for i in best]
    return result


def predict_message(text) -> Prediction:
    """Classify one raw message exactly the way the training data was processed."""
    text = validate_message(text)
    model = load_model()

    scores = model.predict_proba(_to_model_input(text))[0]
    by_label = dict(zip(model.classes_, scores))
    probabilities = {label: float(by_label[label]) for label in config.LABELS}

    label = max(probabilities, key=probabilities.get)
    risk = 1.0 - probabilities["ham"]
    prediction = Prediction(label, probabilities, risk, risk_level(risk))

    logger.info("Predicted %s (risk %.2f, %d chars)", label, risk, len(text))
    return prediction
