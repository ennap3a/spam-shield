"""Spam Shield, page 2: how good is the model, on what data, and where does it fail?

Everything here is read from reports/ (small files saved by the training scripts),
so the page works even where the raw datasets are not available.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from spam_shield import config

st.title("📊 Model analytics")

metrics = ui.load_report("test_metrics.json")
summary = ui.load_report("dataset_summary.json")
challenge = ui.load_report("challenge_results.csv")
challenge_correct = (challenge["label"] == challenge["predicted"]).mean()

# --- Headline numbers -----------------------------------------------------------
st.subheader("Headline results")
st.caption(f"Test set: {metrics['test_messages']:,} messages the model never saw while learning "
           "(used once). Challenge set: 40 hand-written messages from outside the datasets.")
c1, c2, c3, c4 = st.columns(4)
overall, baseline = metrics["overall"], metrics["baseline_always_ham"]
c1.metric("Macro-F1", f"{overall['macro_f1']:.3f}",
          f"{overall['macro_f1'] - baseline['macro_f1']:+.3f} vs. baseline",
          help="Our main metric: the F1 of each label, averaged so every label counts equally. "
               "The baseline always answers 'ham'.")
c2.metric("Phishing recall", f"{overall['phishing_recall']:.1%}",
          help="Share of real phishing messages that were caught.")
c3.metric("Accuracy", f"{overall['accuracy']:.1%}",
          f"baseline {baseline['accuracy']:.1%}", delta_color="off",
          help="Misleading on its own: always answering 'ham' already scores this baseline.")
c4.metric("Challenge set", f"{challenge_correct:.0%}",
          help="Hand-written messages: much harder than the test set.")

# --- Per label + confusion matrix -----------------------------------------------
st.subheader("Per label")
left, right = st.columns([1, 1])

with left:
    rows = [{"label": ui.LABEL_NAMES[label],
             "precision": metrics["per_label"][label]["precision"],
             "recall": metrics["per_label"][label]["recall"],
             "f1": metrics["per_label"][label]["f1-score"],
             "messages": int(metrics["per_label"][label]["support"])}
            for label in config.LABELS]
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "precision": st.column_config.ProgressColumn(
                "Precision", format="%.3f", min_value=0, max_value=1,
                help="When the model says this label, how often is it right?"),
            "recall": st.column_config.ProgressColumn(
                "Recall", format="%.3f", min_value=0, max_value=1,
                help="Of the real messages with this label, how many were caught?"),
            "f1": st.column_config.ProgressColumn("F1", format="%.3f", min_value=0, max_value=1),
            "messages": st.column_config.NumberColumn("Test messages", help="Support"),
        },
    )
    st.caption("Recall reads across a row of the matrix; precision reads down a column.")

with right:
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    row_share = [[count / sum(row) for count in row] for row in matrix]
    names = [ui.LABEL_NAMES[label] for label in labels]
    fig = go.Figure(go.Heatmap(
        z=row_share, x=names, y=names,
        colorscale=ui.SEQUENTIAL_BLUE[ui.theme()], zmin=0, zmax=1, showscale=False,
        xgap=2, ygap=2,
        customdata=matrix,
        hovertemplate="True %{y} → predicted %{x}<br>%{customdata:,} messages "
                      "(%{z:.1%} of the row)<extra></extra>",
    ))
    # Light theme: big shares are dark cells -> white text. Dark theme: the opposite.
    dark_theme = ui.theme() == "dark"
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            strong = row_share[i][j] > 0.5
            ink = "#0b0b0b" if (strong and dark_theme) else (
                "#ffffff" if strong or dark_theme else "#0b0b0b")
            fig.add_annotation(x=names[j], y=names[i], text=f"{count:,}", showarrow=False,
                               font={"color": ink})
    fig.update_yaxes(autorange="reversed", title="True label", showgrid=False)
    fig.update_xaxes(title="Predicted label", side="top", showgrid=False, tickangle=0)
    st.markdown("**Confusion matrix** (color = share of each true label)")
    st.plotly_chart(ui.style_figure(fig, height=340), width="stretch",
                    config={"displayModeBar": False})

# --- How the model was chosen ---------------------------------------------------
st.subheader("How the model was chosen")
st.caption("All choices were made on a validation split of the training data, never on the test set.")
comparison = ui.load_report("model_comparison.csv").rename(columns={"name": "model"})
st.dataframe(
    comparison[["model", "val_macro_f1", "val_phishing_recall", "challenge_macro_f1", "seconds"]],
    hide_index=True, width="stretch",
    column_config={
        "model": "Model",
        "val_macro_f1": st.column_config.ProgressColumn(
            "Validation macro-F1", format="%.3f", min_value=0, max_value=1),
        "val_phishing_recall": st.column_config.ProgressColumn(
            "Validation phishing recall", format="%.3f", min_value=0, max_value=1),
        "challenge_macro_f1": st.column_config.ProgressColumn(
            "Challenge macro-F1", format="%.3f", min_value=0, max_value=1),
        "seconds": st.column_config.NumberColumn("Training time (s)", format="%.0f"),
    },
)
st.caption("LinearSVC was chosen: best validation macro-F1, best challenge score among the "
           "classifiers, and fast. Its C=0.5 and balanced class weights won a 5-fold "
           "cross-validated grid search.")

left, right = st.columns(2)
with left:
    reliability = ui.load_report("reliability.csv")
    color = ui.NEUTRAL_SERIES[ui.theme()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", hoverinfo="skip",
                             line={"color": ui.CHROME[ui.theme()]["muted"], "width": 1,
                                   "dash": "dot"}))
    fig.add_trace(go.Scatter(
        x=reliability["confidence"], y=reliability["accuracy"], mode="lines+markers",
        line={"color": color, "width": 2}, marker={"size": 9, "color": color},
        customdata=reliability["messages"],
        hovertemplate="Model said %{x:.0%} sure → right %{y:.0%} of the time"
                      "<br>%{customdata:,} messages<extra></extra>",
    ))
    fig.update_xaxes(title="How sure the model was", tickformat=".0%", range=[0.3, 1.02])
    fig.update_yaxes(title="How often it was right", tickformat=".0%", range=[0, 1.02])
    st.markdown("**Are the probabilities honest?** (dotted line = perfect)")
    st.plotly_chart(ui.style_figure(fig, height=300), width="stretch",
                    config={"displayModeBar": False})
with right:
    thresholds = ui.load_report("risk_thresholds.csv")
    st.markdown("**Choosing the warning level** (validation data)")
    st.dataframe(
        thresholds, hide_index=True, width="stretch",
        column_config={
            "threshold": st.column_config.NumberColumn("Warn when risk ≥", format="%.1f"),
            "ham_flagged": st.column_config.ProgressColumn(
                "Normal messages warned", format="%.3f", min_value=0, max_value=1),
            "harmful_caught": st.column_config.ProgressColumn(
                "Harmful messages caught", format="%.3f", min_value=0, max_value=1),
        },
    )
    st.caption(f"The app warns from risk {config.SUSPICIOUS_RISK} (Suspicious) and "
               f"{config.HIGH_RISK} (High risk).")

# --- Data ------------------------------------------------------------------------
st.subheader("Training data")
st.caption(f"{summary['total_messages']:,} messages after cleaning "
           f"({summary['train_messages']:,} train / {summary['test_messages']:,} test).")
table = pd.DataFrame(summary["label_by_source"]).reindex(config.LABELS)
table.index = [ui.LABEL_NAMES[label] for label in table.index]
table["median length (chars)"] = [summary["median_chars_by_label"][label] for label in config.LABELS]
st.dataframe(table, width="stretch")
st.caption("⚠️ Each harmful label comes from only one channel: spam only from SMS, phishing "
           "only from email, toxic only from chat. This is the biggest limitation of the project.")

st.subheader("Words the model relies on most")
top_words = ui.load_report("top_words.json")
columns = st.columns(4)
for column, label in zip(columns, config.LABELS):
    words = top_words[label][:12]
    if label == "toxic":
        words = [ui.mask_word(word) for word in words]
    column.markdown(f"**{ui.LABEL_NAMES[label]}**")
    column.markdown("\n".join(f"- `{word}`" for word in words))
st.caption("Some top words reveal where the data came from rather than what the message "
           "means: years (2000–2005) and names in the old email datasets. These are known "
           "leaks, listed in the backlog.")

# --- Challenge set -----------------------------------------------------------------
st.subheader("Hand-written challenge set")
mistakes = challenge[challenge["label"] != challenge["predicted"]]
st.caption(f"{len(challenge) - len(mistakes)} of {len(challenge)} correct. The mistakes show "
           "what the model has not learned yet:")
st.dataframe(
    mistakes.assign(label=mistakes["label"].map(ui.LABEL_NAMES),
                    predicted=mistakes["predicted"].map(ui.LABEL_NAMES)),
    hide_index=True, width="stretch",
    column_config={"text_content": "Message", "label": "True label",
                   "predicted": "Model said", "source": "Channel", "note": "What it tests"},
)

# --- This session ---------------------------------------------------------------------
st.subheader("Your session")
history = st.session_state.get("history", [])
if not history:
    st.caption("Messages you analyze on the Classifier page appear here.")
else:
    history_table = pd.DataFrame(history)
    history_table["label"] = history_table["label"].map(ui.LABEL_NAMES)
    st.dataframe(history_table, hide_index=True, width="stretch",
                 column_config={"message": "Message", "label": "Most likely",
                                "risk_score": st.column_config.ProgressColumn(
                                    "Risk", format="%.2f", min_value=0, max_value=1),
                                "risk_level": "Level"})
