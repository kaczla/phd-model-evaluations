from typing import List

from pydantic import BaseModel


class TokenizedToken(BaseModel):
    raw_token: str
    tokens: List[str]
    indexes: List[int]
