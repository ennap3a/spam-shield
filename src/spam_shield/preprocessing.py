"""Text cleaning for Spam Shield.

Every message passes through clean_text() before it reaches the model,
both during training and when the app makes a prediction.
"""

import re


from functools import lru_cache

import spacy
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS

# --- Regex patterns: compiled once, reused for every message ---
HTML_TAG_PATTERN = re.compile(r"<[a-zA-Z/!][^>]*>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]+")
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
# spaCy only needs the start of a message; this caps memory and time.
MAX_NLP_CHARS = 10_000

# Order matters: the specific forms come before the general "n't",
# otherwise "won't" would become "wo not".
CONTRACTIONS = [
    ("won't", "will not"),
    ("can't", "cannot"),
    ("n't", " not"),
    ("'re", " are"),
    ("'m", " am"),
    ("'ll", " will"),
    ("'ve", " have"),
    ("'d", " would"),
]

# Negations flip meaning ("not spam" vs "spam"), so we never remove them.
NEGATION_WORDS = {"no", "not", "nor", "never", "none", "nothing",
                  "nobody", "neither", "nowhere"}

# Words that are useless only because of where our data came from.
# "enron" appears in ~10% of ham emails: a shortcut, not a real signal.
DOMAIN_STOP_WORDS = {"enron"}

# spaCy's list, minus the negations, plus our domain words.
STOP_WORDS = (SPACY_STOP_WORDS - NEGATION_WORDS) | DOMAIN_STOP_WORDS


def strip_html(text: str) -> str:
    """Replace HTML tags like <p> or </div> with a space."""
    return HTML_TAG_PATTERN.sub(" ", text)


def replace_urls_and_emails(text: str) -> str:
    """Replace links with the token URL and addresses with EMAIL."""
    text = URL_PATTERN.sub(" URL ", text)
    return EMAIL_PATTERN.sub(" EMAIL ", text)


def normalize_apostrophes(text: str) -> str:
    """Turn curly apostrophes (’ ‘) into the plain ' so contractions match."""
    return text.replace("\u2019", "'").replace("\u2018", "'")


def expand_contractions(text: str) -> str:
    """Expand short forms: "can't" -> "cannot", "you're" -> "you are"."""
    for short, full in CONTRACTIONS:
        text = text.replace(short, full)
    return text


def remove_non_ascii(text: str) -> str:
    """Replace non-ASCII characters (emoji, mojibake like ï½) with a space."""
    text = NON_ASCII_PATTERN.sub(" ", text)
    return text
   


def remove_special_characters(text: str) -> str:
    """Keep only lowercase letters, digits and whitespace."""
    return SPECIAL_CHAR_PATTERN.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces, tabs and newlines into one space, trim the ends."""
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()

def clean_text(text: str) -> str:
    """Run the full cleaning pipeline on one raw message."""
    text = strip_html(text)
    text = replace_urls_and_emails(text)
    text = normalize_apostrophes(text)
    text = text.lower()
    text = expand_contractions(text)
    text = remove_non_ascii(text)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)
    return text




@lru_cache(maxsize=1)
def get_nlp():
    """Load the spaCy model once; later calls reuse the same copy."""
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lemmatize_and_filter(doc) -> str:
    """Turn a spaCy doc into a string of lemmas, without stop words."""
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        if token.is_space or lemma in STOP_WORDS:
            continue
        tokens.append(lemma)
    return " ".join(tokens)


def preprocess_texts(texts: list[str], batch_size: int = 1000) -> list[str]:
    """Clean, lemmatize and filter many messages at once (fast)."""
    nlp = get_nlp()
    cleaned = [clean_text(text)[:MAX_NLP_CHARS] for text in texts]
    docs = nlp.pipe(cleaned, batch_size=batch_size)
    return [lemmatize_and_filter(doc) for doc in docs]


def preprocess_text(text: str) -> str:
    """Same as preprocess_texts, for a single message (used by the app later)."""
    return preprocess_texts([text])[0]