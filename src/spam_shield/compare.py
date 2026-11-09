"""Task 5.1: compare classifiers on the validation split and on the challenge set.

Every candidate gets the same features (TF-IDF + metadata) and the same data.
The test set is NOT used here: choosing a model is done on validation only.
"""

import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from spam_shield import config
from spam_shield.challenge import load_challenge_set
from spam_shield.evaluate import compute_metrics
from spam_shield.features import MODEL_METADATA_COLUMNS, build_feature_transformer
from spam_shield.prepare import load_split, split_train_test
from spam_shield.train import build_model

# name: (classifier, use_balanced_sample_weights)
CANDIDATES = {
    "ComplementNB (4.2 winner)": (ComplementNB(), False),
    "LogisticRegression": (
        LogisticRegression(C=10, class_weight="balanced", max_iter=2000),
        False,
    ),
    "LinearSVC": (LinearSVC(C=0.5, class_weight="balanced"), False),
    "XGBoost": (
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.2,
            tree_method="hist",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        True,
    ),
}

# Which metadata columns to give the model (tested with LinearSVC).
METADATA_OPTIONS = {
    "with length (3.3)": ["char_count", "word_count", *MODEL_METADATA_COLUMNS],
    "without length (chosen)": MODEL_METADATA_COLUMNS,
    "no metadata": [],
}


def fit_and_predict(classifier, use_weights, fit_part, datasets, features=None):
    """Train on fit_part, then predict every DataFrame in `datasets`.

    XGBoost only accepts numbers as labels, so for it we translate
    "ham"/"spam"/... into 0/1/... before training and back afterwards.
    """
    model = build_model(classifier, features)
    needs_numbers = isinstance(classifier, XGBClassifier)
    label_to_number = {label: i for i, label in enumerate(config.LABELS)}
    y = fit_part["label"].map(label_to_number) if needs_numbers else fit_part["label"]

    fit_params = {}
    if use_weights:
        fit_params["classifier__sample_weight"] = compute_sample_weight("balanced", y)
    model.fit(fit_part, y, **fit_params)

    predictions = []
    for df in datasets:
        predicted = model.predict(df)
        if needs_numbers:
            predicted = [config.LABELS[i] for i in predicted]
        predictions.append(predicted)
    return predictions


def score_row(name, validation, challenge, val_pred, challenge_pred, seconds) -> dict:
    """One results row: validation and challenge scores side by side."""
    val = compute_metrics(validation["label"], val_pred)
    chal = compute_metrics(challenge["label"], challenge_pred)
    return {
        "name": name,
        "val_macro_f1": val["macro_f1"],
        "val_phishing_recall": val["phishing_recall"],
        "challenge_macro_f1": chal["macro_f1"],
        "challenge_phishing_recall": chal["phishing_recall"],
        "seconds": seconds,
    }


def run(experiments, fit_part, validation, challenge) -> pd.DataFrame:
    """Train and score every experiment: name -> (classifier, use_weights, features)."""
    rows = []
    for name, (classifier, use_weights, features) in experiments.items():
        start = time.time()
        val_pred, challenge_pred = fit_and_predict(
            classifier, use_weights, fit_part, [validation, challenge], features
        )
        rows.append(score_row(name, validation, challenge, val_pred, challenge_pred,
                              time.time() - start))
        print(f"done: {name} ({time.time() - start:.0f}s)")
    return pd.DataFrame(rows).set_index("name").round(3)


if __name__ == "__main__":
    train = load_split(config.TRAIN_FILE)
    fit_part, validation = split_train_test(train, config.TEST_SIZE, config.RANDOM_STATE)
    challenge = load_challenge_set()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. Which metadata columns? (LinearSVC) ===")
    feature_experiments = {
        name: (LinearSVC(C=0.5, class_weight="balanced"), False,
               build_feature_transformer(metadata_columns=columns))
        for name, columns in METADATA_OPTIONS.items()
    }
    features_table = run(feature_experiments, fit_part, validation, challenge)
    print("\n", features_table.to_string())
    features_table.to_csv(config.REPORTS_DIR / "feature_comparison.csv")

    print("\n=== 2. Which classifier? (chosen features) ===")
    model_experiments = {
        name: (classifier, use_weights, None)
        for name, (classifier, use_weights) in CANDIDATES.items()
    }
    models_table = run(model_experiments, fit_part, validation, challenge)
    print("\n", models_table.to_string())
    models_table.to_csv(config.REPORTS_DIR / "model_comparison.csv")
    print(f"\nSaved both tables to {config.REPORTS_DIR}")
