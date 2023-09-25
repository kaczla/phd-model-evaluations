from typing import Optional

from pydantic import BaseModel


class LossStatistics(BaseModel):
    model_name: str
    model_class_name: str
    model_human_name: Optional[str] = None
    join_examples: bool
    sequence_length: int
    loss: float

    def get_model_name(self) -> str:
        if self.model_human_name is not None:
            return self.model_human_name

        return self.model_name
