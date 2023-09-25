from typing import List

from pydantic import BaseModel


class TokenPredictionMLM(BaseModel):
    token_size: int
    prefix_indexes: List[int]
    prefix_tokens: List[str]
    token_indexes: List[int]
    token_str_list: List[str]
