import pandas as pd

from spam_shield.prepare import split_train_test


def make_fake_data() -> pd.DataFrame:
    """40 tiny messages: 10 for each (label, source) group."""
    groups = [("ham", "email"), ("phishing", "email"), ("spam", "sms"), ("toxic", "chat")]
    rows = [
        {"text_content": f"message {label} {i}", "label": label, "source": source}
        for label, source in groups
        for i in range(10)
    ]
    return pd.DataFrame(rows)


def test_split_sizes():
    train_df, test_df = split_train_test(make_fake_data(), test_size=0.2, random_state=42)
    assert len(train_df) == 32
    assert len(test_df) == 8


def test_no_message_in_both_train_and_test():
    train_df, test_df = split_train_test(make_fake_data(), test_size=0.2, random_state=42)
    assert set(train_df["text_content"]).isdisjoint(test_df["text_content"])


def test_every_group_is_in_test():
    train_df, test_df = split_train_test(make_fake_data(), test_size=0.2, random_state=42)
    assert (test_df["label"].value_counts() == 2).all()
