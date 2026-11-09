# Feature Backlog

| Tier | Feature | Purpose | Complexity |
|---|---|---|---|
| Next stage | Explainability (highlight influential words) | Trust & debugging | Medium |
| Next stage | Batch CSV classification | Realistic enterprise use | Low |
| Next stage | Advanced URL analysis (IP URLs, look-alike domains, shorteners) | Better phishing detection | Medium |
| Advanced | DistilBERT transformer comparison | Modern NLP skills | High |
| Advanced | FastAPI REST API + Docker | Production-style service | Medium-High |
| Advanced | Multi-label classification | Messages can be spam AND toxic | Medium |
| Advanced | MLflow experiment tracking | Professional ML workflow | Medium |
| Optional | User feedback → retraining | Human-in-the-loop | Medium |
| Optional | Data drift monitoring | MLOps awareness | High |
| Data fix | Investigate CSV quoting in `load_phishing_emails()`: one broken row swallowed thousands of emails (17M chars); currently dropped by `MAX_MESSAGE_CHARS` | Recover lost emails | Low |
| Data fix | Add email spam (e.g. SpamAssassin) and SMS phishing (smishing) examples: today spam only exists as SMS and phishing only as email | Stop the model learning the channel instead of the label | Medium |
