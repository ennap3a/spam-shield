# Progress Log

## Current status
- **Phase:** 0 – Foundations
- **Next task:** 1.1 – Choose and load datasets

## Completed
- [x] 0.1 – Development environment (Python 3.12.10, venv, libraries, spaCy model)
- [x] 0.2 – Git, GitHub, folder structure, first commit

## Key decisions
| Decision | Choice | Reason |
|---|---|---|
| NLP library | spaCy | Accurate lemmatization without manual grammar tagging |
| Classification type | Single-label (v1) | Simpler; multi-label planned for later |
| Main metric | Macro-F1 + phishing recall | Accuracy is misleading on imbalanced classes |

## What I learned
- Training vs inference, and why preprocessing must be identical
- Virtual environments, pip, requirements.txt, version pinning
- Git: working folder → staging → commit; .gitignore; GitHub remotes