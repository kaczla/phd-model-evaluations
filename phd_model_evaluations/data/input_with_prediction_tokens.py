from typing import List

from pydantic import BaseModel

from phd_model_evaluations.data.prediction_token import PredictionToken


class InputWithPredictionTokens(BaseModel):
    input_indexes: List[int]
    predicted_tokens: List[PredictionToken]
