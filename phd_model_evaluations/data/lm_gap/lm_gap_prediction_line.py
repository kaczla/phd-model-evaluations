from typing import List

from pydantic import BaseModel


class LMGapPredictionLine(BaseModel):
    left_context_token_indexes: List[int]
    right_context_token_indexes: List[int]
