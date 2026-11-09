"""Task 5.3: turn LinearSVC scores into trustworthy probabilities.

LinearSVC only gives a "score", not a percentage. CalibratedClassifierCV learns
how to translate those scores into probabilities. "Calibrated" means: when the
model says 90%, it is right about 90% of the time.

We compare two methods on the validation split:
- sigmoid: fits a smooth S-shaped curve (simple, stable with little data)
- isotonic: fits a flexible step curve (needs more data)

Then we choose the risk levels the app shows. risk = 1 - P(ham), i.e. how
likely the message is NOT a normal message. The thresholds are chosen on
validation, never on the test or challenge set.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.svm import LinearSVC

from spam_shield import config
from spam_shield.evaluate import compute_metrics
from spam_shield.prepare import load_split, split_train_test
from spam_shield.train import build_model


def risk_threshold_table(model, validation: pd.DataFrame) -> pd.DataFrame:
    """For several thresholds: how many normal messages get a warning, and how
    many harmful ones (spam/phishing/toxic) are caught."""
    ham_column = list(model.classes_).index("ham")
    risk = 1 - model.predict_proba(validation)[:, ham_column]
    is_ham = (validation["label"] == "ham").to_numpy()
    rows = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
        rows.append({
            "threshold": threshold,
            "ham_flagged": (risk[is_ham] >= threshold).mean(),
            "harmful_caught": (risk[~is_ham] >= threshold).mean(),
        })
    return pd.DataFrame(rows)


def calibration_error(probabilities: np.ndarray, correct: np.ndarray, bins: int = 10):
    """Expected calibration error + a reliability table.

    Messages are grouped by how confident the model was (0-10%, ..., 90-100%).
    In each group we compare the average confidence with how often it was right.
    A perfectly calibrated model has confidence == accuracy in every group.
    """
    confidence = probabilities.max(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    rows, error = [], 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > low) & (confidence <= high)
        if not in_bin.any():
            continue
        avg_confidence = confidence[in_bin].mean()
        accuracy = correct[in_bin].mean()
        error += in_bin.mean() * abs(avg_confidence - accuracy)
        rows.append({"bin": f"{low:.1f}-{high:.1f}", "confidence": avg_confidence,
                     "accuracy": accuracy, "messages": int(in_bin.sum())})
    return error, pd.DataFrame(rows)


if __name__ == "__main__":
    train = load_split(config.TRAIN_FILE)
    fit_part, validation = split_train_test(train, config.TEST_SIZE, config.RANDOM_STATE)

    summary, reliability_tables = [], {}
    for method in ["sigmoid", "isotonic"]:
        classifier = CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced"), method=method, cv=5
        )
        model = build_model(classifier).fit(fit_part, fit_part["label"])
        probabilities = model.predict_proba(validation)
        predicted = model.classes_[probabilities.argmax(axis=1)]
        correct = predicted == validation["label"].to_numpy()

        error, reliability = calibration_error(probabilities, correct)
        reliability_tables[method] = reliability
        if method == "sigmoid":
            thresholds = risk_threshold_table(model, validation)
        summary.append({
            "method": method,
            "log_loss": log_loss(validation["label"], probabilities, labels=model.classes_),
            "calibration_error": error,
            **compute_metrics(validation["label"], predicted),
        })

    summary = pd.DataFrame(summary).set_index("method").round(4)
    print(summary.to_string())
    print("\nReliability (sigmoid): confidence vs. how often it was right")
    print(reliability_tables["sigmoid"].round(3).to_string(index=False))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.REPORTS_DIR / "calibration_comparison.csv")
    reliability_tables["sigmoid"].to_csv(config.REPORTS_DIR / "reliability.csv", index=False)

    print("\nRisk thresholds (sigmoid): normal messages warned vs harmful caught")
    print(thresholds.round(3).to_string(index=False))
    thresholds.to_csv(config.REPORTS_DIR / "risk_thresholds.csv", index=False)
