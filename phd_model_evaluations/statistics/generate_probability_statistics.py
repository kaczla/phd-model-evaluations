import numpy as np

from phd_model_evaluations.data.statistics.accepted_score_statistics import AcceptedScoreStatistics
from phd_model_evaluations.data.statistics.score_statistics import ScoreStatistics


def generate_score_statistics(scores: np.ndarray) -> ScoreStatistics:
    return ScoreStatistics(
        total_elements=scores.size,
        max_value=scores.max(),
        min_value=scores.min(),
        avg_value=scores.mean(),
        std_value=scores.std(),
    )


def generate_accepted_score_statistics(scores: np.ndarray) -> AcceptedScoreStatistics:
    score_statistics = generate_score_statistics(scores)
    return AcceptedScoreStatistics(
        total_elements=score_statistics.total_elements,
        max_value=score_statistics.max_value,
        min_value=score_statistics.min_value,
        avg_value=score_statistics.avg_value,
        std_value=score_statistics.std_value,
        accepted_min_value=score_statistics.avg_value - score_statistics.std_value,
        accepted_max_value=score_statistics.avg_value + score_statistics.std_value,
    )
