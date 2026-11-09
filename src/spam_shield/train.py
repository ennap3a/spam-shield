"""Train the final model (features + classifier in one Pipeline) and save it to disk."""

import time

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spam_shield import config
from spam_shield.features import build_feature_transformer
from spam_shield.prepare import load_split


def build_model(classifier, features=None) -> Pipeline:
    """Glue the feature machine and a classifier into ONE object."""
    return Pipeline([
        ("features", features if features is not None else build_feature_transformer()),
        ("classifier", classifier),
    ])


def build_final_model() -> Pipeline:
    """The chosen model, with the evidence for each choice.

    - LinearSVC: best validation macro-F1 in Task 5.1 (reports/model_comparison.csv)
    - C=0.5, class_weight="balanced": best 5-fold CV score in Task 5.2
      (reports/tuning_results.csv)
    - CalibratedClassifierCV(sigmoid): turns SVM scores into probabilities
      that match reality, Task 5.3 (reports/calibration_comparison.csv)
    """
    classifier = CalibratedClassifierCV(
        LinearSVC(C=0.5, class_weight="balanced"), method="sigmoid", cv=5
    )
    return build_model(classifier)


if __name__ == "__main__":
    train = load_split(config.TRAIN_FILE)

    model = build_final_model()
    start = time.time()
    model.fit(train, train["label"])
    print(f"Trained on {len(train):,} messages in {time.time() - start:.1f} seconds")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_FILE)
    print(f"Saved model to {config.MODEL_FILE}")
