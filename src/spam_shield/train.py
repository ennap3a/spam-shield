"""Train a model (features + classifier in one Pipeline) and save it to disk."""

import time

import joblib
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from spam_shield import config
from spam_shield.features import build_feature_transformer
from spam_shield.prepare import load_split


def build_model(classifier) -> Pipeline:
    """Glue the feature machine and a classifier into ONE object."""
    return Pipeline([
        ("features", build_feature_transformer()),
        ("classifier", classifier),
    ])


if __name__ == "__main__":
    train = load_split(config.TRAIN_FILE)

    # Winner of Task 4.2 (see notebooks/03_model_experiments.ipynb):
    # best validation macro-F1 (0.866) while keeping ham recall above 0.9.
    model = build_model(ComplementNB())
    start = time.time()
    model.fit(train, train["label"])
    print(f"Trained on {len(train):,} messages in {time.time() - start:.1f} seconds")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_FILE)
    print(f"Saved model to {config.MODEL_FILE}")