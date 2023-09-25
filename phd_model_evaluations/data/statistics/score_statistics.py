from pydantic import BaseModel


class ScoreStatistics(BaseModel):
    total_elements: int
    max_value: float
    min_value: float
    avg_value: float
    std_value: float
