import json
import torch

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "./saved_distilbert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

    with open(f"{MODEL_PATH}/labels.json","r") as file:
        labels = json.load(file)

    return tokenizer, model, labels

@asynccontextmanager
async def lifespan(app: FastAPI):

    tokenizer, model, labels = load_model()

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.labels = labels

    yield


app = FastAPI(
    title="Customer Support Ticket Classifier",
    lifespan=lifespan)


class TicketRequest(BaseModel):

    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):

        if not value.strip():
            raise ValueError("Text cannot be empty")

        return value.strip()


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "model")
    }

@app.post("/predict")
def predict(request: TicketRequest):

    tokenizer = app.state.tokenizer
    model = app.state.model
    labels = app.state.labels

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits,dim=1)

        confidence, prediction = torch.max(probabilities,dim=1)

    category = labels[prediction.item()]

    high_keywords = [
        "urgent",
        "emergency",
        "charged twice",
        "fraud",
        "account hacked"
    ]

    priority = "High" if any(
        word in request.text.lower()
        for word in high_keywords
    ) else "Normal"

    return {
        "category": category,
        "priority": priority,
        "confidence": round(confidence.item(), 4)
    }