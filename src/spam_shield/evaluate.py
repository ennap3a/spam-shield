"""Grade a trained model on the test set: overall, per label, and per source."""

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from spam_shield import config
from spam_shield.prepare import load_split


def compute_metrics(y_true, y_pred) -> dict:
    """The three numbers we care about most."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "phishing_recall": recall_score(y_true, y_pred, labels=["phishing"], average="macro", zero_division=0),
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"{name:<12} " + "  ".join(f"{key}={value:.3f}" for key, value in metrics.items()))


def print_report(test: pd.DataFrame, predictions) -> None:
    """Per-label report, confusion matrix, and results per source."""
    print("\n=== Per label ===")
    print(classification_report(test["label"], predictions, labels=config.LABELS, digits=3, zero_division=0))

    print("=== Confusion matrix (rows = truth, columns = prediction) ===")
    matrix = confusion_matrix(test["label"], predictions, labels=config.LABELS)
    print(pd.DataFrame(matrix, index=config.LABELS, columns=config.LABELS))

    print("\n=== Per source ===")
    results = test.assign(predicted=predictions)
    for source, group in results.groupby("source"):
        print_metrics(source, compute_metrics(group["label"], group["predicted"]))


if __name__ == "__main__":
    train = load_split(config.TRAIN_FILE)
    test = load_split(config.TEST_FILE)

    # The "dumb" baseline: always predicts the most common label (ham).
    dummy = DummyClassifier(strategy="most_frequent").fit(train, train["label"])
    print_metrics("always-ham", compute_metrics(test["label"], dummy.predict(test)))

    model = joblib.load(config.MODEL_FILE)
    predictions = model.predict(test)
    print_metrics("naive bayes", compute_metrics(test["label"], predictions))

    print_report(test, predictions)