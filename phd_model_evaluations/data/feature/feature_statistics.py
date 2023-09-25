import logging
from collections import Counter
from typing import Counter as CounterType

from pydantic import BaseModel

from phd_model_evaluations.data.lm_gap.lm_gap_context_line_with_gap import LMGapContextLineWithGap
from phd_model_evaluations.utils.common_utils import get_key_from_directory_items

LOGGER = logging.getLogger(__name__)


class FeatureStatistics(BaseModel):
    token_frequency: CounterType = Counter()
    sequence_token_length: CounterType = Counter()
    are_aggregated: bool = False

    def update_statistics(self, lm_gap_line: LMGapContextLineWithGap) -> None:
        tokens = [lm_gap_line.gap, *lm_gap_line.left_context.split(), *lm_gap_line.right_context.split()]
        self.token_frequency.update(tokens)
        self.sequence_token_length.update([len(tokens)])

    def aggregate_statistics(self, remove_low_frequency_of_token: bool = True) -> None:
        LOGGER.debug(
            f"Aggregating LM-GAP feature statistics (removing low frequency of tokens: {remove_low_frequency_of_token})"
        )
        self.token_frequency = (
            Counter({k: v for k, v in self.token_frequency.most_common() if v > 1})
            if remove_low_frequency_of_token
            else Counter({k: v for k, v in self.token_frequency.most_common()})
        )
        self.sequence_token_length = Counter(
            {k: v for k, v in sorted(self.sequence_token_length.items(), key=get_key_from_directory_items)}
        )
        self.are_aggregated = True
