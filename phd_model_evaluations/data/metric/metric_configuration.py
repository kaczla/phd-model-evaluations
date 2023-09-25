from typing import Optional

from pydantic import BaseModel


class MetricConfiguration(BaseModel):
    name: str
    configuration_name: Optional[str] = None
