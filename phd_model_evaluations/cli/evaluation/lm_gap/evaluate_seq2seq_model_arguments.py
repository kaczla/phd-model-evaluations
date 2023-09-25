from dataclasses import dataclass, field


@dataclass
class LMGapEvaluateSeq2SeqArguments:
    """Arguments for LM-GAP evaluation for seq2seq (encoder-decoder) models."""

    depth: int = field(
        default=3,
        metadata={
            "help": "How depth do prediction - it's equal to number of tokens to predict."
            " Greater size will increase prediction time and allows to generate longer tokens."
        },
    )
    top_k_minimum: int = field(
        default=5,
        metadata={"help": "Minimum number of predictions in next/deeper predictions."},
    )
    top_k_reduce: int = field(default=3, metadata={"help": "Number of reduced top-k for next/deper predictions."})
