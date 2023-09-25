from enum import Enum


class VisualizationType(str, Enum):
    histogram = "histogram"
    pie = "pie"

    def __str__(self) -> str:
        return self.value
