from pydantic import BaseModel


class PredictionBase(BaseModel):
    token: str
    score: float
