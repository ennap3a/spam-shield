"""Spam Shield API: lets other programs use the model over HTTP."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from spam_shield import config
from spam_shield.predict import InvalidMessageError, explain_prediction, predict_message

app = FastAPI(
    title="Spam Shield API",
    description="Classifies a message as ham, spam, phishing or toxic.",
    version="1.1.0",
)

BAD_MESSAGE = {400: {"description": "A message is empty or too long."}}


# ---------- Forms (what comes in and what goes out) ----------


class MessageIn(BaseModel):
    """One message sent by the client."""

    text: str = Field(examples=["Congratulations! You won a FREE iPhone, click here to claim"])


class BatchIn(BaseModel):
    """Many messages sent in one request."""

    texts: list[str] = Field(
        min_length=1,
        max_length=config.MAX_BATCH_SIZE,
        examples=[["Are we still meeting for lunch?", "WIN a FREE iPhone!!! Reply YES now"]],
    )


class PredictionOut(BaseModel):
    """The result for one message."""

    label: str
    probabilities: dict[str, float]
    risk_score: float
    risk_level: str
    risk_message: str


class BatchOut(BaseModel):
    """The results for many messages, in the same order they were sent."""

    results: list[PredictionOut]


class WordWeight(BaseModel):
    """One word and how much it pushed towards the label."""

    word: str
    weight: float


class ExplanationOut(BaseModel):
    """The label plus the words that pushed the model towards it."""

    label: str
    words: list[WordWeight]


# ---------- Helper (the shared "kitchen station") ----------


def classify(text: str) -> PredictionOut:
    """Run the model on one message, turning bad input into a 400 error."""
    try:
        result = predict_message(text)
    except InvalidMessageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return PredictionOut(
        label=result.label,
        probabilities=result.probabilities,
        risk_score=result.risk_score,
        risk_level=result.risk_level,
        risk_message=result.risk_message,
    )


# ---------- Endpoints (the menu) ----------


@app.get("/")
def home():
    return {"message": "Welcome to the Spam Shield API! Go to /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/labels")
def labels():
    return {"labels": config.LABELS}


@app.post("/predict", response_model=PredictionOut, responses=BAD_MESSAGE)
def predict(message: MessageIn):
    return classify(message.text)


@app.post("/predict/batch", response_model=BatchOut, responses=BAD_MESSAGE)
def predict_batch(batch: BatchIn):
    results = []
    for number, text in enumerate(batch.texts, start=1):
        try:
            results.append(classify(text))
        except HTTPException as error:
            raise HTTPException(
                status_code=400, detail=f"Message {number}: {error.detail}"
            ) from error
    return BatchOut(results=results)


@app.post("/explain", response_model=ExplanationOut, responses=BAD_MESSAGE)
def explain(message: MessageIn):
    prediction = classify(message.text)
    words = explain_prediction(message.text, prediction.label)
    return ExplanationOut(
        label=prediction.label,
        words=[WordWeight(word=word, weight=round(weight, 4)) for word, weight in words],
    )
