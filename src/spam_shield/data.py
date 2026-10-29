"""Load the raw datasets and convert each one to our common format.

Every loader returns a DataFrame with exactly three columns:
    text_content -- the raw message text
    label        -- one of config.LABELS ("ham", "spam", "phishing", "toxic")
    source       -- the channel the message came from ("sms", "email", "chat")
"""

import csv
import pandas as pd
from spam_shield import config
OUTPUT_COLUMNS =["text_content", "label", "source"]

def load_sms_spam(path=config.SMS_SPAM_FILE) -> pd.DataFrame:
    df =pd.read_csv(
        path,
        sep="\t", #columns separated by TAB
        header=None, # the file has no header 
        names=["label","text_content"], #we name the columns ourselves
        quoting=csv.QUOTE_NONE,  #treat quote mark as normal text
        encoding="utf-8"
    
                    )
    df["source"] = "sms"   # create a new column; every row gets "sms" as the source that the data came from
    return df[OUTPUT_COLUMNS]


def load_phishing_emails(path=config.PHISHING_FILE) -> pd.DataFrame:

    df=pd.read_csv(path) #read the csv of phishing emails
    df=df.rename(columns={"Email Text":"text_content"}) #translate or change their label to our default label
    label_map ={"Safe Email": "ham", "Phishing Email": "phishing"}
    df["label"] = df["Email Type"].map(label_map)
    df["source"]= "email"
    return df[OUTPUT_COLUMNS]

def load_toxic_comments(path=config.TOXIC_FILE) ->pd.DataFrame:
    df=pd.read_csv(path)
    df=df.rename(columns={"comment_text":"text_content"})
    toxicity_columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    is_toxic=df[toxicity_columns].sum(axis=1)>0
    df["label"]=is_toxic.map({True:"toxic", False:"ham"})

    df["source"]="chat"
    return df[OUTPUT_COLUMNS]

def combine_datasets(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack several datasets (with identical columns) into one."""
    return pd.concat(frames, ignore_index=True)


def drop_empty_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose text is missing or contains only whitespace.  """

    df = df.dropna(subset=["text_content"]).copy() # to drop the empty messages
    df["text_content"]= df["text_content"].str.strip() # to remove the extra spaces that might exist after the end of the message
    #like "WIN a prize" ---> "WIN a prize "
    df= df[df["text_content"] !=""]

    return df



def drop_duplicate_messages(df: pd.DataFrame)->pd.DataFrame:
    """Remove repeated messages and messages that appear with conflicting labels"""
    df=df.drop_duplicates(subset=["text_content","label"])
    is_conflicting=df.duplicated(subset=["text_content"], keep=False)
    return df[~is_conflicting]


def sample_chat_messages(df: pd.DataFrame, per_class: int, random_state: int) -> pd.DataFrame:
    """Randomly keep `per_class` rows of each label from the chat source only."""
    is_chat = df["source"] == "chat"
    chat_sample = (
        df[is_chat]
        .groupby("label")                                    # split chat rows by label
        .sample(n=per_class, random_state=random_state)      # pick N random rows from EACH group
    )
    return pd.concat([df[~is_chat], chat_sample], ignore_index=True)



def validate_dataset(df: pd.DataFrame) -> None:
    """Stop with a clear error if the dataset breaks any of our rules."""
    assert df["text_content"].notna().all(), "Found missing text"
    assert not df["text_content"].duplicated().any(), "Found duplicate texts"
    unexpected = set(df["label"]) - set(config.LABELS)
    assert not unexpected, f"Unexpected labels: {unexpected}"
    assert df["message_id"].is_unique, "message_id values are not unique"

def print_summary(name: str, df:pd.DataFrame) ->None:
    print(f"\n=={name}: {len(df):,} rows==")
    print(df["label"].value_counts(dropna=False))


def build_dataset() -> pd.DataFrame:
    """Run the full pipeline: load → combine → clean → dedupe → balance → validate."""
    df = combine_datasets([load_sms_spam(), load_phishing_emails(), load_toxic_comments()])
    print(f"Combined:            {len(df):>8,} rows")

    df = drop_empty_messages(df)
    print(f"After dropping empty:{len(df):>8,} rows")

    df = drop_duplicate_messages(df)
    print(f"After deduplication: {len(df):>8,} rows")

    df = sample_chat_messages(df, config.CHAT_SAMPLE_PER_CLASS, config.RANDOM_STATE)
    print(f"After chat sampling: {len(df):>8,} rows")

    df.insert(0, "message_id", range(1, len(df) + 1))   # IDs 1, 2, 3, ... as the FIRST column
    validate_dataset(df)
    return df


def save_dataset(df: pd.DataFrame, path=config.PROCESSED_DATASET_FILE) -> None:
    """Save the dataset as CSV, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df):,} rows to {path}")


if __name__ == "__main__":
    dataset = build_dataset()
    save_dataset(dataset)

    print_summary("Final dataset", dataset)
    print("\nLabels per source:")
    print(pd.crosstab(dataset["label"], dataset["source"]))









