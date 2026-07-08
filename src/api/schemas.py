"""
Pydantic models defining the API's response shapes.

Defining these explicitly (instead of returning raw dicts) gives FastAPI
automatic request/response validation AND automatic OpenAPI docs at /docs —
an interviewer opening your API's /docs page and seeing well-typed responses
looks a lot more professional than an undocumented endpoint.
"""

from typing import List

from pydantic import BaseModel


class ClassPrediction(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_predictions: List[ClassPrediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
