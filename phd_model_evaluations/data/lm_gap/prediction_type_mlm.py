from enum import Enum


class PredictionTypeMLM(Enum):
    simple = "simple"
    loss = "loss"
    mixed = "mixed"

    def __str__(self) -> str:
        return str(self.value)
