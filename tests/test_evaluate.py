from spam_shield.evaluate import compute_metrics


def test_perfect_predictions():
    truth = ["ham", "ham", "phishing", "phishing"]
    metrics = compute_metrics(truth, truth)
    assert metrics == {"accuracy": 1.0, "macro_f1": 1.0, "phishing_recall": 1.0}


def test_half_of_phishing_caught():
    truth = ["ham", "ham", "phishing", "phishing"]
    predicted = ["ham", "phishing", "phishing", "ham"]
    metrics = compute_metrics(truth, predicted)
    assert metrics["accuracy"] == 0.5
    assert metrics["phishing_recall"] == 0.5     # 1 of the 2 real phishing caught
