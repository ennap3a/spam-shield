# Spam Shield

A machine learning and NLP system that classifies messages (email, SMS, chat)
as **Legitimate, Spam, Phishing, or Toxic**, with real-time risk scores in an
interactive Streamlit dashboard.

> 🚧 **Status:** In development. Phase 0 (project setup).

## Setup

```bash
git clone https://github.com/<ennap3a>/spam-shield.git
cd spam-shield
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Project structure

| Folder | Purpose |
|---|---|
| `data/raw/` | Original downloaded datasets (not committed) |
| `data/processed/` | Cleaned, unified dataset (not committed) |
| `notebooks/` | Data exploration and experiments |
| `src/spam_shield/` | Main source code (cleaning, features, training, prediction) |
| `app/` | Streamlit web application |
| `models/` | Saved trained models |
| `tests/` | Automated tests |