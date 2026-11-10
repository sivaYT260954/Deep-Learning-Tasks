Customer Support Ticket Classifier

Project Description

This project uses DistilBERT and FastAPI to classify customer support tickets based on category and priority.

Installation

Install the required dependencies:

python -m pip install -r requirements.txt

Run Application

Run the application from the project folder:

uvicorn main:app --reload

API Documentation

GET /health
POST /predict

API Documentation:

http://127.0.0.1:8000/docs

Health Check

Method: GET

Endpoint: "/health"

Prediction

Method: POST

Endpoint: "/predict"

Request:

{
    "text": "I was charged twice"
}

Example Response:

{
    "category": "Billing",
    "priority": "High",
    "confidence": 0.93
}

Run Tests

python -m pytest -v

