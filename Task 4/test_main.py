import torch
import pytest

from types import SimpleNamespace
from fastapi.testclient import TestClient

import main

class FakeTokenizer:

    def __call__(self, text, **kwargs):

        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]])
        }


class FakeModel:

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):

        return SimpleNamespace(
            logits=torch.tensor([[0.1, 2.0, 0.2]])
        )


@pytest.fixture
def client(monkeypatch):

    def fake_load_model():

        return (
            FakeTokenizer(),
            FakeModel(),
            ["billing", "refund", "technical"]
        )

    monkeypatch.setattr(main,"load_model",fake_load_model)

    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_text(client):

    response = client.post("/predict",json={"text": "I want a refund"})

    assert response.status_code == 200
    assert response.json()["category"] == "refund"
    assert response.json()["priority"] == "Normal"
    assert "confidence" in response.json()


def test_empty_text(client):

    response = client.post("/predict",json={"text": ""})
    assert response.status_code == 422

def test_missing_text(client):

    response = client.post("/predict",json={})

    assert response.status_code == 422


def test_whitespace_text(client):

    response = client.post("/predict",json={"text": " "})

    assert response.status_code == 422