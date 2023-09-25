from pydantic import BaseModel


class LMGapResults(BaseModel):
    model_name: str
    score: float
