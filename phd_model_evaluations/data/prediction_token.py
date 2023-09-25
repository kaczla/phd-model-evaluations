from typing import List

from pydantic import BaseModel


class PredictionToken(BaseModel):
    token_indexes: List[int]
    scores: List[float]
    total_score: float

    def get_current_depth(self) -> int:
        return len(self.token_indexes)

    def create_next_prediction_token(self, token_index: int, score: float) -> "PredictionToken":
        return PredictionToken(
            token_indexes=self.token_indexes + [token_index],
            scores=self.scores + [score],
            total_score=self.total_score * score,
        )
