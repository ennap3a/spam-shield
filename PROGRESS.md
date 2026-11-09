# Progress Log

## Current status
- **Phase:** 9 – Done (v1). Next: deploy to Streamlit Cloud and add the live link to README
- **Next task:** see BACKLOG.md

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
- [x] 4.3 – Hand-written challenge set (40 messages) + error analysis
- [x] 5.1 – Compared LR, LinearSVC, XGBoost; removed length features (they reveal the source)
- [x] 5.2 – 5-fold cross-validated grid search for LinearSVC
- [x] 5.3 – Sigmoid calibration, risk thresholds chosen on validation, final model + reports
- [x] 6 – `predict.py` service layer: input validation, risk levels, word explanations
- [x] 7 – Streamlit app: classifier page + analytics dashboard
- [x] 8 – Logging, ruff, GitHub Actions CI (fresh-install check passes)
- [x] 9 – README with results and limitations, learning guide, deploy instructions

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
| Model (Task 4.2, replaced in 5.1) | ComplementNB | Best validation macro-F1 (0.866) with ham recall 0.911. Balanced weights gave higher phishing recall (0.961) but ham recall fell to 0.772: about 1 in 4 normal messages flagged |

| Features (Task 5.1) | No `char_count` / `word_count` | Length reveals the channel; removing it kept validation macro-F1 (0.911 → 0.905) and improved the challenge set (0.495 → 0.528) |
| Final model (5.1–5.3) | LinearSVC(C=0.5, balanced) + sigmoid calibration | Best validation macro-F1; best of 12 settings in 5-fold CV (0.911 ± 0.005); calibration error 1.3% |
| Risk levels (5.3) | Suspicious ≥ 0.3, High ≥ 0.7 | At 0.3 on validation: 94% of harmful messages warned, 10% of normal ones |
| Model file | Committed to git | So the app runs on Streamlit Cloud without the datasets |

## Results
| Model | Data | Macro-F1 | Phishing recall | Accuracy |
|---|---|---|---|---|
| Always "ham" | test | 0.189 | 0.000 | 0.607 |
| MultinomialNB | test | 0.636 | 0.752 | 0.864 |
| ComplementNB | validation | 0.866 | 0.912 | 0.896 |
| ComplementNB | test | 0.871 | 0.901 | 0.896 |
| ComplementNB | challenge set (40) | 0.535 | 0.600 | 0.575 |
| LinearSVC (calibrated) | 5-fold CV on train | 0.911 | – | – |
| **LinearSVC (calibrated), final** | **test** | **0.918** | **0.918** | **0.928** |
| LinearSVC (calibrated), final | challenge set (40) | 0.495 | 0.200 | 0.525 |

## Challenge set findings (Task 4.3)
- **Spam vs phishing is really "SMS vs email".** In our data, spam only exists as SMS and phishing only as email. 6 of 10 spam messages written as emails were called phishing.
- **Phishing without a link or with a new trick is missed.** The IT-helpdesk password request, the CEO gift-card scam and the Netflix payment text were all called ham.
- **Short casual messages are unstable.** Casual chat ham was called toxic, and short insults with no swear words were called ham. This matches the test set, where 13% of normal SMS were called toxic.
- **Money words push towards phishing.** A friend asking to split a bill was called phishing.
- Next: check whether stronger models (Phase 5) make the same mistakes; consider adding email spam and SMS phishing data (see BACKLOG).

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
- A high test score can hide label problems: the challenge set showed our labels are tied to the source
- Stronger models can be MORE overfit to the dataset style: better validation, worse on new messages
- Cross-validation (5 practice exams) and calibration (probabilities that mean what they say)
- A service layer keeps the app simple: it only calls predict_message()
- CI: every push is linted and tested on a fresh machine
