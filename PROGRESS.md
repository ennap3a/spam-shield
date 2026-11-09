# Progress Log

## Current status
- **Phase:** 4 – First model + honest evaluation
- **Next task:** 4.3 – Hand-written challenge set + error analysis

## Completed
- [x] 0.1 – Development environment (Python 3.12, venv, libraries, spaCy model)
- [x] 0.2 – Git, GitHub, folder structure, first commit
- [x] 0.3 – Moved from Windows to Mac (rebuilt venv, fixed git email and .gitignore)
- [x] 1.1–1.2 – Loaded SMS, phishing and toxic datasets into one common format
- [x] 1.3 – EDA notebook: found the `enron` artifact and source leakage
- [x] 2.1 – Text cleaning module (regex) with unit tests
- [x] 2.2 – spaCy lemmatization and stop words (negations kept, `enron` removed)
- [x] 3.1 – One-time preprocessing + stratified train/test split + TF-IDF exploration
- [x] 3.2 – Fixed leaks (spaced URLs, `wikipedia`), raw-text metadata features
- [x] 3.3 – One feature transformer: TF-IDF + scaled metadata
- [x] 4.1 – First model (MultinomialNB) vs "always-ham" baseline
- [x] 4.2 – Compared imbalance fixes on a validation split, chose ComplementNB

## Key decisions
| Decision | Choice | Reason |
|---|---|---|
| NLP library | spaCy | Accurate lemmatization without manual grammar tagging |
| Classification type | Single-label (v1) | Simpler; multi-label planned for later |
| Main metric | Macro-F1 + phishing recall | Accuracy is misleading on imbalanced classes |
| Overlong messages | Drop rows > 50,000 chars | One broken CSV row was 17M chars (thousands of emails glued together) |
| Domain stop words | `enron`, `wikipedia` | They reveal where a message came from, not what kind it is |
| `uppercase_ratio` | Computed but not used by the model | The email dataset was lowercased by its creators, so it reveals the source |
| Train/test split | 80/20, stratified by label + source | Every group keeps the same share in train and test |
| Model selection | Compare on a validation split from train; test set used once | Choosing with the test set would make its score dishonest |
| Model (Task 4.2) | ComplementNB | Best validation macro-F1 (0.866) with ham recall 0.911. Balanced weights gave higher phishing recall (0.961) but ham recall fell to 0.772: about 1 in 4 normal messages flagged |

## Results
| Model | Data | Macro-F1 | Phishing recall | Accuracy |
|---|---|---|---|---|
| Always "ham" | test | 0.189 | 0.000 | 0.607 |
| MultinomialNB | test | 0.636 | 0.752 | 0.864 |
| ComplementNB | validation | 0.866 | 0.912 | 0.896 |
| ComplementNB | test | 0.871 | 0.901 | 0.896 |

## What I learned
- Training vs inference, and why preprocessing must be identical
- Virtual environments, pip, requirements.txt, version pinning
- Git: working folder → staging → commit; .gitignore; GitHub remotes
- A venv belongs to one machine and records its own path: move the project first, then rebuild it
- Regex, unit tests with pytest, and why the order of pipeline steps matters
- Data has bugs too: one broken row only showed up when spaCy refused it
- Leakage: words and features that reveal the source of a message instead of its label
- fit on train only, transform on test; a test value above 1 after scaling is proof of no peeking
- Accuracy hides small labels; macro-F1, recall, precision, support and the confusion matrix don't
- Validation set = practice exam, test set = real exam (used once)
