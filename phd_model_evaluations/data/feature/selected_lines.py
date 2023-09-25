from typing import List

from pydantic import BaseModel


class SelectedLines(BaseModel):
    source_set_name: str
    source_total_lines: int
    total_lines: int
    total_lines_percentage: float
    indexes: List[int]
