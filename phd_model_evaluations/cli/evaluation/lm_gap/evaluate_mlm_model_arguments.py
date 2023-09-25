from dataclasses import dataclass, field

from phd_model_evaluations.data.lm_gap.prediction_type_mlm import PredictionTypeMLM


@dataclass
class LMGapEvaluateMLMArguments:
    """Arguments for LM-GAP evaluation for MLM (encoder) models."""

    method: PredictionTypeMLM = field(metadata={"help": "Type of prediction method."})
    max_token_size: int = field(
        default=1,
        metadata={
            "help": "Number of tokens used in prediction, greater size will increase prediction time"
            " and allows to generate longer tokens."
        },
    )

    def __post_init__(self) -> None:
        self.method = PredictionTypeMLM[str(self.method)]
