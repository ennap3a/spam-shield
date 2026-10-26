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

def print_summary(name: str, df:pd.DataFrame) ->None:
    print(f"\n=={name}: {len(df):,} rows==")
    print(df["label"].value_counts(dropna=False))


if __name__ == "__main__":
    print_summary("SMS spam", load_sms_spam())
    print_summary("Phishing emails", load_phishing_emails())
    print_summary("Toxic comments", load_toxic_comments())









