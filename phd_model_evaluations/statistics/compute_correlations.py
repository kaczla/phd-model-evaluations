import logging
from itertools import product
from typing import Dict, List, Tuple

from scipy.stats import pearsonr, spearmanr

from phd_model_evaluations.data.results.aggregated_results import AggregatedResults
from phd_model_evaluations.data.statistics.correlations.computed_correlations import ComputedCorrelations
from phd_model_evaluations.data.statistics.correlations.correlation_data import CorrelationData

LOGGER = logging.getLogger(__name__)


def get_reversed_result_sign(value: float, reverse_result_sign: bool) -> float:
    if reverse_result_sign:
        return -1.0 * value

    return value


def compute_correlations(
    aggregated_results: AggregatedResults,
    x_labels: List[str],
    y_labels: List[str],
    x_higher_better: bool,
    y_higher_better: bool,
    save_values: bool = False,
) -> List[ComputedCorrelations]:
    LOGGER.info(f"Computing correlation for X labels: {', '.join(x_labels)} and Y labels: {', '.join(y_labels)}...")

    reverse_result_sign = (x_higher_better and not y_higher_better) or (not x_higher_better and y_higher_better)
    LOGGER.info(
        f"X higher value is better: {x_higher_better}, Y higher value is better: {y_higher_better},"
        f" reserve result sign: {reverse_result_sign}"
    )

    correlations = []
    for x_label, y_label in product(x_labels, y_labels):
        x_results = aggregated_results.results[x_label]
        y_results = aggregated_results.results[y_label]
        x_values, y_values = [], []
        for model_name in aggregated_results.model_list:
            x = x_results.get(model_name)
            y = y_results.get(model_name)
            if x is None or y is None:
                continue

            x_values.append(x)
            y_values.append(y)
        # Compute correlation
        pearson_correlation = pearsonr(x_values, y_values)
        spearman_correlation = spearmanr(x_values, y_values)
        computed_correlations = ComputedCorrelations(
            x_name=x_label,
            y_name=y_label,
            pearson_correlation=CorrelationData(
                correlation=get_reversed_result_sign(pearson_correlation.statistic, reverse_result_sign),
                p_value=pearson_correlation.pvalue,
            ),
            spearman_correlation=CorrelationData(
                correlation=get_reversed_result_sign(spearman_correlation.correlation, reverse_result_sign),
                p_value=spearman_correlation.pvalue,
            ),
            x_higher_better=x_higher_better,
            y_higher_better=y_higher_better,
            x_values=x_values if save_values else None,
            y_values=y_values if save_values else None,
        )
        LOGGER.info(f"Computed correlation for {x_label} and {y_label} with {len(x_values)} values")
        correlations.append(computed_correlations)

    LOGGER.info(f"Computed {len(correlations)} correlations")
    return correlations


def add_previous_correlations(
    correlations: List[ComputedCorrelations], previous_correlations: List[ComputedCorrelations]
) -> List[ComputedCorrelations]:
    previous_correlations_map: Dict[Tuple[str, str], ComputedCorrelations] = {
        (correlation.x_name, correlation.y_name): correlation for correlation in previous_correlations
    }

    total_merged = 0
    for correlation in correlations:
        previous_correlation = previous_correlations_map.get((correlation.x_name, correlation.y_name))
        if previous_correlation is not None:
            total_merged += 1
            correlation.previous_pearson_correlation = previous_correlation.pearson_correlation
            correlation.previous_spearman_correlation = previous_correlation.spearman_correlation

    if total_merged == len(correlations):
        LOGGER.info(f"Total merged {total_merged} correlations")
    else:
        LOGGER.info(f"Total merged {total_merged} correlations from {len(correlations)}")

    return correlations
