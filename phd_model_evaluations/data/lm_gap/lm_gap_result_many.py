from typing import Dict, List

from pydantic import BaseModel

from phd_model_evaluations.data.lm_gap.lm_gap_result import LMGapResult


class LMGapResultMany(BaseModel):
    name: str
    model_names: List[str]
    model_name_to_result: Dict[str, LMGapResult]
