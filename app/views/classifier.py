"""Spam Shield, page 1: classify a message."""

import streamlit as st

import ui
from spam_shield.predict import (
    InvalidMessageError,
    explain_prediction,
    load_model,
    predict_message,
)

EXAMPLES = {
    "Choose an example…": "",
    "Prize spam (SMS)": "WINNER!! You have been selected for a £1000 prize. Text CLAIM to 80086 now!",
    "Bank phishing (email)": (
        "Dear customer, your account has been suspended due to unusual activity. "
        "Verify your password within 24 hours at http://secure-bank-verify.com"
    ),
    "Normal message": "Hey, are we still meeting for lunch tomorrow at 1? I can book the table.",
    "Toxic comment": "You are a stupid idiot and nobody wants you here.",
}
RISK_STYLE = {
    "safe": (st.success, "✅", "Safe"),
    "suspicious": (st.warning, "⚠️", "Suspicious"),
    "high": (st.error, "⛔", "High risk"),
}


@st.cache_resource
def warm_up_model():
    """Load the model and spaCy once per server, not on every click."""
    load_model()
    predict_message("warm up")
    return True


def use_example():
    st.session_state["message"] = EXAMPLES[st.session_state["example"]]


st.title("🛡️ Spam Shield")
st.caption("Paste an email, SMS or chat message. The model checks whether it is "
           "normal, spam, phishing or toxic, and explains why.")

with st.spinner("Loading the model…"):
    warm_up_model()

st.selectbox("Try an example", list(EXAMPLES), key="example", on_change=use_example)
text = st.text_area("Message", key="message", height=160,
                    placeholder="Paste a message here…")

if st.button("Analyze message", type="primary"):
    try:
        prediction = predict_message(text)
    except InvalidMessageError as error:
        st.warning(f"⚠️ {error}")
        st.stop()

    show_box, icon, level_name = RISK_STYLE[prediction.risk_level]
    show_box(f"{icon} **{level_name}** — {prediction.risk_message}")

    left, right = st.columns(2)
    left.metric("Most likely", ui.LABEL_NAMES[prediction.label])
    right.metric("Risk score", f"{prediction.risk_score:.0%}",
                 help="Risk = 1 − probability of a normal message.")

    st.markdown("**Probability for each label**")
    st.plotly_chart(ui.probability_chart(prediction.probabilities), width="stretch",
                    config={"displayModeBar": False})

    words = explain_prediction(text, prediction.label)
    st.markdown(f"**Words that pushed towards “{ui.LABEL_NAMES[prediction.label]}”**")
    if words:
        shown = [ui.mask_word(w) if prediction.label == "toxic" else w for w, _ in words]
        st.markdown(" · ".join(f"`{word}`" for word in shown))
    else:
        st.caption("No single word stood out; the decision came from the message as a whole.")

    st.session_state.setdefault("history", []).append({
        "message": text.strip()[:80],
        "label": prediction.label,
        "risk_score": round(prediction.risk_score, 3),
        "risk_level": prediction.risk_level,
    })

with st.expander("How does it work, and what are its limits?"):
    st.markdown(
        """
1. The message is cleaned (links become `url`, contractions expanded, lowercase).
2. spaCy turns words into their dictionary form and removes filler words, but keeps *not* and *never*.
3. TF-IDF scores the words, plus a few counts: links, email addresses, `!`, currency signs and digits.
4. A calibrated Linear SVM gives a probability for each label.

**Limits, measured honestly:** the model scores 0.92 macro-F1 on held-out test messages,
but only about 0.5 on 40 hand-written messages from outside the training data.
New scam styles (gift-card scams, phishing without a link) are often missed, and spam
written as an email is often called phishing, because in the training data spam only
came from SMS. The *Suspicious* warning exists to catch some of these cases.
See the **Analytics** page for details.
"""
    )
