"""Task 5.2: tune the winning model (LinearSVC) with 5-fold cross-validation.

Cross-validation = 5 practice exams instead of 1. The train set is cut into
5 parts; each part is the "validation" once while the other 4 are used to
learn. The average of the 5 scores is much more reliable than one score.
The test set is still not touched.
"""

import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC

from spam_shield import config
from spam_shield.prepare import load_split
from spam_shield.train import build_model

# Every combination of these values is tried (3 x 2 x 2 = 12 settings).
PARAM_GRID = {
    "classifier__C": [0.1, 0.5, 2.0],                          # how strictly to fit
    "classifier__class_weight": [None, "balanced"],            # boost rare labels?
    "features__tfidf__ngram_range": [(1, 1), (1, 2)],          # words, or words + pairs
}


def build_search() -> GridSearchCV:
    """A grid search over PARAM_GRID, scored with our main metric (macro-F1)."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    return GridSearchCV(
        build_model(LinearSVC()),
        PARAM_GRID,
        scoring="f1_macro",
        cv=folds,
        n_jobs=-1,          # use all CPU cores: folds run in parallel
    )


if __name__ == "__main__":
    train = load_split(config.TRAIN_FILE)

    search = build_search()
    start = time.time()
    search.fit(train, train["label"])
    print(f"Tried {len(search.cv_results_['params'])} settings x 5 folds "
          f"in {time.time() - start:.0f} seconds\n")

    results = pd.DataFrame(search.cv_results_)
    table = (
        results[["param_classifier__C", "param_classifier__class_weight",
                 "param_features__tfidf__ngram_range", "mean_test_score", "std_test_score"]]
        .rename(columns={
            "param_classifier__C": "C",
            "param_classifier__class_weight": "class_weight",
            "param_features__tfidf__ngram_range": "ngram_range",
            "mean_test_score": "cv_macro_f1",
            "std_test_score": "cv_std",
        })
        .sort_values("cv_macro_f1", ascending=False)
        .round(4)
    )
    print(table.to_string(index=False))
    print(f"\nBest settings: {search.best_params_}")
    print(f"Best CV macro-F1: {search.best_score_:.4f}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.REPORTS_DIR / "tuning_results.csv", index=False)
