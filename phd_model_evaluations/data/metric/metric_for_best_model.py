from pydantic import BaseModel


class MetricForBestModel(BaseModel):
    name: str
    greater_is_better: bool
