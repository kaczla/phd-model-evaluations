from pydantic import BaseModel


class LMGapContextLine(BaseModel):
    left_context: str
    right_context: str
