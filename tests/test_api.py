import pytest
from fastapi.testclient import TestClient

from spam_shield import config
from spam_shield.api import app

client = TestClient(app)


def test_health_says_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_labels_come_from_config():
    response = client.get("/labels")

    assert response.status_code == 200
    assert response.json()["labels"] == config.LABELS


def test_predict_normal_message():
    message = {"text": "Hey, are we still meeting for lunch tomorrow at 1?"}

    response = client.post("/predict", json=message)

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "ham"
    assert body["risk_level"] == "safe"
    assert set(body["probabilities"]) == set(config.LABELS)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_predict_rejects_empty_message(text):
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 400
    assert response.json()["detail"] == "The message is empty."


def test_predict_rejects_too_long_message():
    text = "x" * (config.MAX_MESSAGE_CHARS + 1)
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 400
    assert "too long" in response.json()["detail"]


def test_predict_obvious_spam_is_high_risk():
    message = {"text": "WIN a FREE iPhone!!! Reply YES to claim your prize now"}

    response = client.post("/predict", json=message)

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "spam"
    assert body["risk_level"] == "high"


def test_predict_rejects_wrong_body():
    wrong_name = client.post("/predict", json={"message": "hi"})
    assert wrong_name.status_code == 422

    wrong_type = client.post("/predict", json={"text": 123})
    assert wrong_type.status_code == 422


def test_batch_returns_one_result_per_message():
    messages = ["Are we still meeting for lunch?", "WIN a FREE iPhone!!! Reply YES now", "Thanks!"]

    response = client.post("/predict/batch", json={"texts": messages})

    assert response.status_code == 200
    assert len(response.json()["results"]) == len(messages)


def test_batch_tells_which_message_is_bad():
    response = client.post("/predict/batch", json={"texts": ["hello", "   ", "bye"]})

    assert response.status_code == 400
    assert "Message 2" in response.json()["detail"]


def test_batch_rejects_too_many_messages():
    too_many = ["hi"] * (config.MAX_BATCH_SIZE + 1)

    response = client.post("/predict/batch", json={"texts": too_many})

    assert response.status_code == 422


def test_explain_spam_shows_spam_words():
    message = {"text": "WIN a FREE iPhone!!! Reply YES to claim your prize now"}

    response = client.post("/explain", json=message)

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "spam"
    words = [item["word"] for item in body["words"]]
    assert "prize" in words
