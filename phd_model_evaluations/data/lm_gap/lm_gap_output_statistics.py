from pydantic import BaseModel

from phd_model_evaluations.data.statistics.positions_statistics import PositionsStatistics
from phd_model_evaluations.data.statistics.score_statistics import ScoreStatistics


class LMGapOutputStatistics(BaseModel):
    total_lines: int
    total_probabilities_not_summed_to_one: int
    total_token_match: int
    total_token_not_match: int
    unk_statistics: ScoreStatistics
    first_token_statistics: ScoreStatistics
    token_match_positions: PositionsStatistics
