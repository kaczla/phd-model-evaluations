from enum import Enum


class LatexTableType(str, Enum):
    TABLE = "TABLE"
    SIDEWAYSTABLE = "SIDEWAYSTABLE"
    LONGTABLE = "LONGTABLE"

    def __str__(self) -> str:
        return str(self.value)
