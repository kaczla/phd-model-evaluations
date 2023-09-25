from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LMGapEvaluateArguments:
    """Arguments for LM-GAP evaluation shared between method scripts."""

    file_in: Path = field(metadata={"help": "Input file path."})
    file_out: Path = field(metadata={"help": "Output file path."})
    column_left_context: int = field(
        default=1,
        metadata={"help": "Number of column of left context in input file, indexed from 0."},
    )
    column_right_context: int = field(
        default=2,
        metadata={"help": "Number of column of left context in input file, indexed from 0."},
    )
    top_k: int = field(
        default=15,
        metadata={"help": "Number of predictions saved in top-k predictions data (.jsonl file)."},
    )
    unk_probability: float = field(
        default=0.001,
        metadata={
            "help": "Probability for UNK token, this value will be set in output and make sure"
            " that probability in the each line is summed to 1.0."
        },
    )
    smart_tokens: bool = field(
        default=False,
        metadata={
            "help": "Reduce number of tokens to predict by smart selecting sub-tokens,"
            " e.g. it will select token which can produce whole word - it will"
            " ignore sub-token with the space/punctuation in the middle of the word."
        },
    )
    generate_out_file_name: bool = field(
        default=False,
        metadata={"help": "Generate output file name, base on given arguments."},
    )
    save_aggregations: bool = field(
        default=False,
        metadata={
            "help": "Save all raw predicted tokens to further aggregations."
            "It will save in `aggregation_*.jsonl` file, where `*` is the name of output file path."
        },
    )
    overwrite_output: bool = field(
        default=False, metadata={"help": "Overwrite output file, will not raise exception if output file exists"}
    )
