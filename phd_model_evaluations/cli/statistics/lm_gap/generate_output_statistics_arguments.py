from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LMGapOutputStatisticsArguments:
    """Arguments for generate statistics for LM-GAP predictions."""

    path: Path = field(metadata={"help": "Directory path of LM-GAP predictions."})
