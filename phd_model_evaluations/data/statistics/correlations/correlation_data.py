from typing import Optional

from pydantic import BaseModel


class CorrelationData(BaseModel):
    correlation: float
    p_value: Optional[float] = None
