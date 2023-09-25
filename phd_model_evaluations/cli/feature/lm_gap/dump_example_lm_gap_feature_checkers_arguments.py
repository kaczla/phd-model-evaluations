from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DumpExampleLMGapFeatureCheckersArguments:
    """Arguments for dumping example LM-GAP feature checkers."""

    save_path: Path = field(
        metadata={
            "help": "Path to example configuration of LM-GAP feature checkers,"
            " it will save example configuration for easier changing of LM-GAP feature checkers configuration."
        },
    )
