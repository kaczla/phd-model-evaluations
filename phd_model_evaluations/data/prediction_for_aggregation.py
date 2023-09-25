from typing import List

from pydantic import BaseModel, Field

from phd_model_evaluations.data.prediction_base import PredictionBase


class PredictionForAggregation(BaseModel):
    type_score: str = Field(regex=r"loss|probability")
    predictions: List[PredictionBase]
