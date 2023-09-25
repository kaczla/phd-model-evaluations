from pydantic import BaseModel, Field


class LMGapResult(BaseModel):
    model_name: str
    score: float = Field(alias="PerplexityHashed")
