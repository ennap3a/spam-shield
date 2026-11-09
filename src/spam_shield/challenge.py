"""Grade the model on hand-written messages from outside our three datasets."""

import joblib
import pandas as pd

from spam_shield import config
from spam_shield.evaluate import compute_metrics, print_metrics
from spam_shield.prepare import add_processed_text


def load_challenge_set(path=config.CHALLENGE_FILE) -> pd.DataFrame:
    """Read the hand-written messages and preprocess them like the training data."""
    return add_processed_text(pd.read_csv(path))


if __name__ == "__main__":
    challenge = load_challenge_set()
    model = joblib.load(config.MODEL_FILE)
    challenge["predicted"] = model.predict(challenge)

    print_metrics("challenge", compute_metrics(challenge["label"], challenge["predicted"]))
    print("\n=== Confusion matrix (rows = truth, columns = prediction) ===")
    print(pd.crosstab(challenge["label"], challenge["predicted"]))

    mistakes = challenge[challenge["label"] != challenge["predicted"]]
    print(f"\n=== {len(mistakes)} mistakes ===")
    for row in mistakes.itertuples():
        print(f"[{row.label} -> {row.predicted}] ({row.note}) {row.text_content}")
