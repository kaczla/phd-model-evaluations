from typing import List, Tuple

from pydantic import BaseModel


class PositionsStatistics(BaseModel):
    most_common_positions: List[int]
    aggregated_positions: List[Tuple[str, int]]
