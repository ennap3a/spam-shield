"""Spam Shield API: lets other programs use the model over HTTP."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from spam_shield import config
from spam_shield.predict import InvalidMessageError, predict_message

app = FastAPI(
    title="Spam Shield API",
    description="Classifies a message as ham, spam, phishing, or toxic.",
    version="1.0.0",
)


class MessageIn(BaseModel):
    """What the client sends."""

    text: str = Field(examples=["Congratulations! You won a FREE iPhone, click here to claim"])


class PredictionOut(BaseModel):
    """What the API sends back."""

    label: str
    probabilities: dict[str, float]
    risk_score: float
    risk_level: str
    risk_message: str


@app.get("/")
def home():
    return {"message": "Welcome to the Spam Shield API! Go to /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/labels")
def labels():
    return {"labels": config.LABELS}


@app.post(
    "/predict",
    response_model=PredictionOut,
    responses={400: {"description": "The message is empty or too long."}},
)
def predict(message: MessageIn):
    try:
        result = predict_message(message.text)
    except InvalidMessageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return PredictionOut(
        label=result.label,
        probabilities=result.probabilities,
        risk_score=result.risk_score,
        risk_level=result.risk_level,
        risk_message=result.risk_message,
    )

