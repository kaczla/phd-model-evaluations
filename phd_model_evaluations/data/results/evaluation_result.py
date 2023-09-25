from typing import Dict, Optional

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    dataset_name: str
    set_name: str
    test_file: str
    model_name: str
    model_human_name: Optional[str]
    best_metric: float
    metrics: Dict[str, float]
