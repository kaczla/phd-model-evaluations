from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LMGapChallengeArguments:
    """Arguments for creating LM-GAP challenge."""

    dataset_name: str = field(metadata={"help": "Name of data sets from `datasets` library."})
    save_path: Path = field(metadata={"help": "A file containing the training data."})
    seed: int = field(default=321, metadata={"help": "Random seed for reproduction."})
    join_sample_text: bool = field(
        default=False,
        metadata={
            "help": "Join text of sample into one text, if disabled then will return each text of sample"
            " into separated line."
        },
    )
    skip_source_test_set: bool = field(
        default=False,
        metadata={
            "help": "Skip source test set and use `validation set` as `test set`"
            " and use part of `train set` as `validation set`"
        },
    )
    overwrite: bool = field(default=False, metadata={"help": "Overwrite challenge data in save directory path."})
