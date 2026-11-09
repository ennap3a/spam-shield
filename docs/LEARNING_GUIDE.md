# Spam Shield: Learning Guide (Phases 5–9)

This guide explains Phases 5–9 of the project. Read one section at a time, open the
file it talks about next to it, and answer the questions at the end of each section
before you look at the answers.

**How to use it:** for each section, (1) read "The idea", (2) read the code in the file,
(3) run the command yourself and compare with the numbers here, (4) answer the questions.

---

## The whole project on one page

```
data.py        3 datasets → one table (text_content, label, source), cleaned and balanced
prepare.py     spaCy once → processed_text; split 80% train / 20% test (stratified)
features.py    TF-IDF on processed_text + 5 metadata numbers on the raw text
compare.py     which features? which classifier?  (validation split, never test)
tune.py        which settings?                      (5-fold cross-validation)
calibrate.py   honest probabilities + risk levels   (validation split)
train.py       train the final Pipeline on all of train → models/spam_shield_model.joblib
evaluate.py    the real exam, ONCE → reports/test_metrics.json
challenge.py   40 hand-written messages → reports/challenge_results.csv
predict.py     the only door the app uses: predict_message(text)
app/           Streamlit: Classifier page + Analytics page
.github/       CI: ruff + pytest on every push
```

The golden rule behind the whole order: **decide on train/validation, measure on test once.**

---

## Phase 5.1: Comparing stronger models (`src/spam_shield/compare.py`)

### The idea
Naive Bayes was our first model. Now we try three others on **exactly the same features
and data**, and compare them on the validation split (the practice exam):

| Model | In simple words |
|---|---|
| **Logistic Regression** | Gives every word a weight (+ pushes towards a label, − pushes away), adds them up, turns the sum into a probability. |
| **Linear SVM (LinearSVC)** | Also gives every word a weight, but chooses the weights to leave the **widest possible gap** between labels. *Analogy:* drawing the road between two villages as wide as possible, so new houses near the edge are still on the right side. |
| **XGBoost** | Hundreds of small decision trees ("if the message contains *prize* and has more than 2 digits…"); each new tree fixes the mistakes of the previous ones. Very strong on tables, slower on 50,000 sparse word columns. |

### What we found
| Model | Validation macro-F1 | Challenge macro-F1 | Time |
|---|---|---|---|
| ComplementNB | 0.864 | 0.512 | 5 s |
| Logistic Regression | 0.901 | 0.523 | 7 s |
| **LinearSVC** | **0.905** | **0.528** | 5 s |
| XGBoost | 0.884 | 0.516 | 81 s |

**The surprising lesson:** with the length features included, the stronger models scored
*better* on validation but *worse* on the challenge set (LR caught only 1 of 10 hand-written
phishing messages). A stronger model learns the dataset's habits more precisely, including
the bad habits. So we tested which **features** cause it:

| Metadata given to LinearSVC | Validation | Challenge |
|---|---|---|
| with length (`char_count`, `word_count`) | 0.911 | 0.495 |
| **without length (chosen)** | 0.905 | 0.528 |
| no metadata at all | 0.905 | 0.549 |

Length mostly says "this is an email" (long) or "this is an SMS" (short), which is the
channel leak again. We removed it. "No metadata" is 1 message better out of 40, which is
noise, so we kept the 5 meaningful counts (links, email addresses, `!`, currency, digits).

### Code to read
- `fit_and_predict()`: XGBoost only accepts numbers as labels, so `"ham"` → `0`, etc., and back.
- `run()`: one loop for every experiment, returning a table. Same code for both tables.
- In `features.py`, `build_feature_transformer(metadata_columns=...)` lets an experiment
  choose the columns. `FunctionTransformer(..., kw_args={"columns": ...})` passes them on.
- In `train.py`, `build_model(classifier, features=None)` accepts a custom feature machine.

### Questions
1. Why do we compare models on validation and not on the test set?
2. Why can a *better* validation score come with a *worse* challenge score?
3. Why did we keep the metadata even though "no metadata" scored 0.549 vs 0.528?

<details><summary>Answers</summary>

1. If we pick the model that is best on the test set, we used the exam to make a decision,
   and its score is no longer an honest estimate. Validation is the practice exam.
2. Validation comes from the same datasets as training, so learning dataset habits (length,
   style, years) is rewarded there. The challenge set has none of those habits, so the same
   learning hurts.
3. 0.021 on 40 messages is a difference of about one message: noise. The 5 counts are real
   signals in real life (scams do contain links and money).
</details>

---

## Phase 5.2: Cross-validation tuning (`src/spam_shield/tune.py`)

### The idea
A model has **settings** we choose (hyperparameters), for example LinearSVC's `C`:

- **small C** → "keep the weights small and simple, ignore a few odd training messages"
- **big C** → "fit every training message as well as possible" (risk: memorizing)

One validation split is one practice exam. Luck plays a role. **5-fold cross-validation**
cuts train into 5 parts; each part is the practice exam once while the other 4 are used to
learn. The average of 5 scores is much more reliable.

```
fold 1: [VAL][ fit ][ fit ][ fit ][ fit ]
fold 2: [ fit ][VAL][ fit ][ fit ][ fit ]
 ...                                      → average the 5 scores
fold 5: [ fit ][ fit ][ fit ][ fit ][VAL]
```

**GridSearchCV** tries every combination in `PARAM_GRID` (3 × 2 × 2 = 12 settings × 5 folds
= 60 trainings) and keeps the best average. `n_jobs=-1` runs them in parallel on all CPU cores.

**Why `classifier__C`?** Same double-underscore rule as `classifier__sample_weight`: "send `C`
to the step called `classifier`". `features__tfidf__ngram_range` goes two levels deep: the
`features` step → its `tfidf` part → the `ngram_range` setting.

### Result
Best: `C=0.5`, `class_weight="balanced"`, words + word pairs → **0.911 ± 0.005**.
The small ± (standard deviation across folds) means the score is stable, not lucky.

### Questions
1. Why is the average of 5 folds better than one validation split?
2. What does `class_weight="balanced"` do, and which earlier task did the same thing?

<details><summary>Answers</summary>

1. Every message is used for validation exactly once, so one lucky or unlucky split cannot
   decide; we also see how much the score varies (the ±).
2. It makes mistakes on rare labels cost more, like the balanced sample weights in Task 4.2.
</details>

---

## Phase 5.3: Honest probabilities (`src/spam_shield/calibrate.py`)

### The idea
The app should say "phishing, 85%". But LinearSVC only gives a **score** (how far from the
road), not a percentage. **Calibration** learns how to translate scores into probabilities.

**Calibrated** means: of all messages where the model says "90% sure", about 90% are right.
*Analogy:* a weather forecaster who says "70% rain" should see rain on about 7 of 10 such days.

`CalibratedClassifierCV(LinearSVC(...), method="sigmoid", cv=5)` trains 5 SVMs (again with
folds) and fits an S-shaped curve from score to probability on the part each SVM did not see.

We measure it with the **reliability table**: group messages by confidence (50–60%, …,
90–100%) and compare with how often they were right. Our sigmoid model: 85.8% confident →
86.9% right; 97.4% → 98.5%. Calibration error: **1.3%**. (Isotonic was equal; sigmoid is
simpler and more stable for small labels like spam.)

### Risk levels
`risk = 1 − P(ham)` = how likely the message is NOT normal. The thresholds were chosen on
validation (`reports/risk_thresholds.csv`):

| Warn when risk ≥ | Normal messages warned | Harmful messages caught |
|---|---|---|
| 0.2 | 13.4% | 95.9% |
| **0.3 (Suspicious)** | **9.7%** | **93.9%** |
| 0.5 | 4.9% | 88.6% |
| **0.7 (High risk)** | 2.7% | 81.1% |

This is the precision/recall trade-off from Task 4.1, turned into a product decision. The
**Suspicious** level helps exactly where the model is weak: on the challenge set, the label
caught 2 of 10 phishing messages, but the Suspicious warning covers 6 of 10.

### Questions
1. Why can't we just use LinearSVC's score as a percentage?
2. Why were the thresholds chosen on validation and not on the challenge set?

<details><summary>Answers</summary>

1. The score has no fixed scale (it can be −3.2 or 1.7); it says which side and how far, not
   how often the model is right. Calibration maps it to real frequencies.
2. Choosing on the challenge set would "train" on it, and it would stop being an honest check.
</details>

---

## Final model and evaluation (`train.py`, `evaluate.py`, `challenge.py`)

`build_final_model()` documents every choice with its evidence in the docstring. Then:

| | Macro-F1 | Phishing recall |
|---|---|---|
| Test set (used once) | **0.918** | **0.918** |
| Challenge set | 0.495 | 0.200 |

The scripts also save small files to `reports/`. **Why?** The dashboard runs on Streamlit
Cloud, where `data/` does not exist (it is not on GitHub). So everything the dashboard shows
is computed once, saved as JSON/CSV, and committed.

New pandas lesson from a bug we hit: `pd.concat([train, test])` keeps both sets of row
numbers, so the index has duplicates and `pd.crosstab` fails. `ignore_index=True` renumbers.

The model's **top words** (`reports/top_words.json`) revealed two more leaks: years
(2000, 2001, 2002…) and names (`vince`, `paliourg`) from the old email datasets. They are in
the backlog.

---

## Phase 6: The service layer (`src/spam_shield/predict.py`)

### The idea
The app should not know about pandas, spaCy, TF-IDF or the model file. It calls **one
function**: `predict_message(text)`. *Analogy:* a restaurant: the customer talks to the
waiter, never to the kitchen. If we change the recipe (the model), the menu stays the same.

### Code to read, top to bottom
- **`InvalidMessageError`**: our own error type, so the app can show a friendly message.
  `class InvalidMessageError(ValueError)` means "a special kind of ValueError".
- **`@dataclass(frozen=True) class Prediction`**: a small box with named fields (`label`,
  `probabilities`, `risk_score`, `risk_level`). `frozen=True` = it cannot be changed after
  creation. `@property risk_message` = a value computed on demand, used like a field.
- **`load_model()`** with `@lru_cache`: load the 10 MB model once, reuse it (same idea as
  `get_nlp()` for spaCy). It raises a clear error if the file is missing.
- **`validate_message()`**: not text → error; empty after `.strip()` → error; longer than
  `MAX_MESSAGE_CHARS` → error. **Never trust input.**
- **`predict_message()`**: builds the same two columns the model was trained on
  (`text_content` raw + `processed_text`), gets the probabilities, orders them by
  `config.LABELS`, computes the risk.
- **`explain_prediction()`**: for each word in *this* message,
  `contribution = TF-IDF value × word weight`. The biggest positive ones are "the words that
  pushed towards this label". The weights are the average over the 5 calibrated SVMs.
- **`logger.info(...)`**: every prediction is logged (label, risk, length), never the
  message text itself, which is private.

### Tests (`tests/test_predict.py`)
- `@pytest.mark.parametrize` runs one test with many inputs: `"", "   ", None, 42` all
  must be rejected. One function, 4 test cases.
- `pytest.raises(InvalidMessageError, match="too long")` checks that the right error,
  with the right message, is raised.
- The risk-level test checks the **edges**: 0.29 → safe, 0.3 → suspicious, 0.69, 0.7 → high.
  Bugs love edges (`>` vs `>=`).

### Questions
1. Why does the app call `predict_message()` instead of `model.predict_proba()` directly?
2. Why do we log the label and length but not the message?

<details><summary>Answers</summary>

1. One place does validation, preprocessing and formatting; the app stays simple, and
   changing the model later only touches `predict.py`.
2. Messages can contain private information; logs are often stored and read by others.
</details>

---

## Phase 7: The Streamlit app (`app/`)

### Structure
```
app/streamlit_app.py     entry point: logging, page config, navigation between 2 pages
app/ui.py                shared helpers: imports, label colors, chart style, report loading
app/views/classifier.py  page 1: type a message → result
app/views/analytics.py   page 2: dashboard from reports/
```

### Streamlit in 5 ideas
1. **The script runs top to bottom on every click.** There is no "event loop" to write.
2. **`st.session_state`** remembers values between runs (the text box, the history list).
   It is shared between pages, which is how the Analytics page shows "Your session".
3. **`@st.cache_resource`** keeps one heavy object (the model) for the whole server;
   **`@st.cache_data`** caches data like the report files.
4. **Widgets with `key=`** store their value in `session_state`. The example dropdown uses
   `on_change=use_example` to copy the example text into the text box's key.
5. **`st.navigation([...])`** defines the pages and their names in the sidebar.

### Design choices
- **One color per label everywhere** (ham blue, spam orange, phishing green, toxic yellow),
  with light and dark versions. The risk box uses status styles (green/yellow/red box)
  **plus an icon and a word**, so meaning never depends on color alone.
- **Probabilities as horizontal bars** with the percentage written at the end of each bar.
- **Confusion matrix as a heatmap**: color = share of the true label (so each row reads
  like recall), the number = count. The text color switches between light and dark
  cells so it stays readable.
- **Offensive words are masked** on screen (`i***t`) by `mask_word()`.
- **`sys.path.insert(0, ".../src")`** in `ui.py` lets `import spam_shield` work on
  Streamlit Cloud without `pip install -e .`.
- **Honesty on the page**: the "How does it work" box and the Analytics page show the
  challenge-set results and the known leaks, not just the good test score.

### Try it
```bash
streamlit run app/streamlit_app.py
```
Paste the gift-card scam from the challenge set. The label says "Normal" but the box says
**Suspicious** (risk ≈ 48%). That is the risk-level design working.

---

## Phase 8: Quality (`pyproject.toml`, `.github/workflows/ci.yml`)

### Logging
`logging.basicConfig(...)` in the app sets the format once; every module uses
`logger = logging.getLogger(__name__)`. *Why not `print`?* Logs have a level (INFO,
WARNING, ERROR), a time and the module name, and can be switched on/off or sent to a file.

### ruff (code style)
`ruff check .` reads every Python file and reports problems. Our rules (`select` in
`pyproject.toml`): **E/W** style, **F** real bugs (unused imports, undefined names),
**I** import order, **B** likely bugs, **UP** modern syntax. It found 40 issues;
`ruff check . --fix` fixed 23 automatically (missing newlines, import order, trailing spaces).
The rest we fixed by hand:
- lines over 100 characters → split them;
- `zip(a, b)` → `zip(a, b, strict=True)`: if `a` and `b` ever have different lengths,
  Python raises an error instead of silently dropping the extra items.

### CI (GitHub Actions)
On every push, GitHub starts a fresh Linux computer, installs **only** what is in
`requirements.txt`, runs `ruff check .` and `pytest -v`. A green check ✅ next to your commit
on GitHub means it all passed; a red ❌ means something broke, with the log one click away.

Two things made this possible:
- The spaCy model is now a line in `requirements.txt` (installed from its official URL),
  so a fresh machine gets it automatically.
- The final model is committed (`.gitignore` has `!models/spam_shield_model.joblib`), so
  the prediction tests can run without the datasets.

We tested this locally first: a brand-new virtual environment, install from
`requirements.txt`, then ruff + 41 tests: all passed.

**A mistake worth remembering:** the first version of the `.gitignore` exception had a
comment on the same line (`!models/spam_shield_model.joblib    # comment`), so git ignored
the rule. Comments in `.gitignore` must be on their own line (the same rule from Phase 0).

---

## Phase 9: Ship it

### README
A recruiter reads the top 20 lines. Ours has: what it does, screenshots, a results table
with a **baseline**, the honest challenge score, how it works, the decisions with evidence,
**limitations**, how to run, how to deploy. Showing the limitations is a strength: it proves
you understand your model.

### Deploy to Streamlit Community Cloud (you do this, it needs your GitHub login)
1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → repo `ennap3a/spam-shield`, branch `main`, file `app/streamlit_app.py`.
3. **Advanced settings** → Python **3.12** → **Deploy**. The first start takes a few minutes
   (it installs everything in `requirements.txt`).
4. Copy the app URL into the README ("Live demo" line), commit, push.

### For your CV
> **Spam Shield**: NLP classifier for spam, phishing and toxic messages (Python,
> scikit-learn, spaCy, Streamlit). Built an end-to-end pipeline over 38k messages from 3
> sources; calibrated Linear SVM reaches 0.92 macro-F1 (baseline 0.19). Found and fixed
> 5 data leaks (dataset artifacts, formatting, length), designed a hand-written challenge
> set that exposed a channel bias, and added a validated "suspicious" risk tier.
> CI with ruff and 41 pytest tests; deployed on Streamlit Cloud.

### Interview questions you can now answer
1. Why macro-F1 and not accuracy? *(always-ham gets 61% accuracy, 0.19 macro-F1)*
2. How did you avoid data leakage? *(fit on train only; selection on validation/CV; test once;
   removed source-revealing words and features)*
3. Your test score is 0.92, why is the challenge score 0.5? *(labels tied to channel; old
   data; it measures generalization, not memorization)*
4. Why calibrate? *(the app shows percentages; they must mean what they say)*
5. What would you do next? *(more varied data: email spam, SMS phishing, modern scams;
   then a transformer such as DistilBERT; see BACKLOG.md)*
