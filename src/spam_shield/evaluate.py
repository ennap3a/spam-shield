"""Grade a trained model on the test set: overall, per label, and per source."""

import json

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
from spam_shield.predict import top_words_per_label
from spam_shield.prepare import load_split


def compute_metrics(y_true, y_pred) -> dict:
    """The three numbers we care about most."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "phishing_recall": recall_score(
            y_true, y_pred, labels=["phishing"], average="macro", zero_division=0
        ),
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"{name:<12} " + "  ".join(f"{key}={value:.3f}" for key, value in metrics.items()))


def build_report(test: pd.DataFrame, predictions, baseline: dict) -> dict:
    """Everything the analytics page shows, as plain numbers (saved as JSON)."""
    results = test.assign(predicted=predictions)
    return {
        "test_messages": len(test),
        "overall": compute_metrics(test["label"], predictions),
        "baseline_always_ham": baseline,
        "per_label": classification_report(
            test["label"], predictions, labels=config.LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": {
            "labels": config.LABELS,
            "matrix": confusion_matrix(test["label"], predictions, labels=config.LABELS).tolist(),
        },
        "per_source": {
            source: compute_metrics(group["label"], group["predicted"])
            for source, group in results.groupby("source")
        },
    }


def print_report(test: pd.DataFrame, predictions) -> None:
    """Per-label report, confusion matrix, and results per source."""
    print("\n=== Per label ===")
    print(classification_report(
        test["label"], predictions, labels=config.LABELS, digits=3, zero_division=0
    ))

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
    baseline = compute_metrics(test["label"], dummy.predict(test))
    print_metrics("always-ham", baseline)

    model = joblib.load(config.MODEL_FILE)
    predictions = model.predict(test)
    print_metrics("final model", compute_metrics(test["label"], predictions))

    print_report(test, predictions)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = config.REPORTS_DIR / "test_metrics.json"
    report_file.write_text(json.dumps(build_report(test, predictions, baseline), indent=2))
    print(f"\nSaved report to {report_file}")

    # Small summaries for the dashboard (the app cannot read data/, it is not on GitHub).
    everything = pd.concat([train, test], ignore_index=True)
    summary = {
        "total_messages": len(everything),
        "train_messages": len(train),
        "test_messages": len(test),
        "label_by_source": pd.crosstab(everything["label"], everything["source"]).to_dict(),
        "median_chars_by_label": (
            everything.groupby("label")["text_content"].apply(lambda t: int(t.str.len().median()))
            .to_dict()
        ),
    }
    (config.REPORTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    (config.REPORTS_DIR / "top_words.json").write_text(
        json.dumps(top_words_per_label(model), indent=2)
    )
    print(f"Saved dataset_summary.json and top_words.json to {config.REPORTS_DIR}")
