"""Spam Shield: the Streamlit entry point. It only sets up logging and the two pages.

Run locally with:  streamlit run app/streamlit_app.py
"""

import logging

import streamlit as st

import ui  # noqa: F401  (adds src/ to the import path for the pages)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Spam Shield", page_icon="🛡️", layout="wide")

page = st.navigation([
    st.Page("views/classifier.py", title="Classifier", icon="🛡️", default=True),
    st.Page("views/analytics.py", title="Analytics", icon="📊"),
])
page.run()
