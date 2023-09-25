from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ComputeLossArguments:
    """Arguments for computing loss."""

    file_path: Path = field(metadata={"help": "Input text file path."})
    save_path: Optional[Path] = field(default=None, metadata={"help": "Save file path."})
    join_examples: bool = field(
        default=False,
        metadata={
            "help": "Join all text examples into one. Each example will be separated by the BOS and EOS."
            " Similar to joining the text as in Casual Language Modeling."
        },
    )
