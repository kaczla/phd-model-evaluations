from typing import Optional

from pydantic import BaseModel


class EmptyDatasetResult(BaseModel):
    dataset_name: str

    def get_score(self, score_factor: float = 1.0) -> Optional[float]:  # noqa: ARG002
        return None
