from spam_shield.preprocessing import (
    clean_text,
    normalize_whitespace,
    preprocess_text,
    strip_html,
)

def test_full_example():
    raw = "<p>Hi!! I CAN'T log in. Visit https://evil.co/login NOW</p>"
    assert clean_text(raw) == "hi i cannot log in visit url now"


def test_html_tags_removed_but_less_than_kept():
    assert clean_text("<b>Free</b> prize") == "free prize"
    assert strip_html("3 < 5") == "3 < 5"


def test_urls_and_emails_become_tokens():
    raw = "Email me at bob@example.com or see www.site.org"
    assert clean_text(raw) == "email me at email or see url"


def test_contractions_expanded():
    raw = "I won't go, you're late, they've left"
    assert clean_text(raw) == "i will not go you are late they have left"


def test_curly_apostrophe():
    assert clean_text("I don\u2019t know") == "i do not know"


def test_numbers_kept():
    assert clean_text("Call 0800 123 NOW!!!") == "call 0800 123 now"

def test_mojibake_removed():
    assert clean_text("ï½") == ""
    assert clean_text("ï½ï½") == ""
    assert clean_text("ï½abc") == "abc"

def test_whitespace_normalized():
    assert normalize_whitespace("  a  b\nc\t") == "a b c"   

def test_message_containing_only_special_characters():
    assert clean_text("!@#$%^&*()") == ""

def test_message_containing_emoji():
    assert clean_text("😀🔥")==""       


def test_lemmatization_and_stop_words():
    assert preprocess_text("The cats were running to the shops") == "cat run shop"


def test_negation_is_kept():
    assert preprocess_text("This is NOT spam") == "not spam"


def test_enron_is_removed():
    assert preprocess_text("Meeting notes from Enron team") == "meeting note team"

def test_spam_like_message():
    assert preprocess_text("Congratulations! 😀🔥 You have WON a FREE prize !!! Visit https://hackes.co now!! ") == "congratulation win free prize visit url"


def test_very_long_message_does_not_crash():
    huge = "free prize " * 500_000      # about 5.5 million characters
    result = preprocess_text(huge)
    assert result.startswith("free prize")

def test_spaced_url_becomes_token():
    assert clean_text("Visit http : / / evil.com / login now")=="visit url now"
