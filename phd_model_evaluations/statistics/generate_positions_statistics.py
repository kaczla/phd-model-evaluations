from collections import Counter
from typing import List, Tuple

from phd_model_evaluations.data.statistics.positions_statistics import PositionsStatistics
from phd_model_evaluations.utils.common_utils import get_key_from_directory_items


def get_aggregate_positions(positions: Counter, max_positions: List[int]) -> List[Tuple[str, int]]:
    sorted_positions = sorted(positions.items(), key=get_key_from_directory_items, reverse=False)
    max_position = sorted_positions[-1][0]

    if max_position not in max_positions:
        max_positions.append(max_position)
        max_positions = sorted(max_positions, reverse=False)

    aggregated_positions = []
    for i_max_position in max_positions:
        if i_max_position > max_position:
            break

        total_occurrences = sum(
            [occurrences for position, occurrences in sorted_positions if position <= i_max_position]
        )
        aggregated_positions.append((f"<={i_max_position}", total_occurrences))
    return aggregated_positions


def get_average_position(positions: Counter) -> float:
    total_elements = float(len(positions))
    sum_positions = float(sum(position * occurrences for position, occurrences in positions.items()))
    return sum_positions / total_elements


def get_positions_statistics(positions: Counter, max_positions: List[int]) -> PositionsStatistics:
    return PositionsStatistics(
        most_common_positions=[position for position, _ in positions.most_common(10)],
        aggregated_positions=get_aggregate_positions(positions, max_positions),
    )
