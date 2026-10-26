"""Central configuration: file paths and label names used across the project."""

from pathlib import Path

# --- Project folders ---
# __file__ is the path of THIS file: .../spam-shield/src/spam_shield/config.py
# .parents[0] = spam_shield folder, [1] = src folder, [2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Raw dataset files ---
SMS_SPAM_FILE = RAW_DATA_DIR / "sms_spam" / "SMSSpamCollection"
PHISHING_FILE = RAW_DATA_DIR / "phishing" / "Phishing_Email.csv"
TOXIC_FILE = RAW_DATA_DIR / "toxic" / "train.csv"

# --- The four classes our model predicts ---
LABELS = ["ham", "spam", "phishing", "toxic"]