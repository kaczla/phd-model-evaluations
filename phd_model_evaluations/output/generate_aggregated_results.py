import logging
from typing import Dict, List, Optional

from phd_model_evaluations.data.results.aggregated_results import AggregatedResults
from phd_model_evaluations.data.results.lm_gap_results import LMGapResults
from phd_model_evaluations.data.statistics.loss_statistics import LossStatistics
from phd_model_evaluations.utils.common_utils import round_float

LOGGER = logging.getLogger(__name__)


def compute_average_score(results_data: AggregatedResults, score_precision: Optional[int] = None) -> AggregatedResults:
    LOGGER.info("Adding average score")
    if "AVG" not in results_data.metrics:
        results_data.metrics["AVG"] = "Average score"

    model_name_to_scores: Dict[str, List[float]] = {}
    for data_set_name, data_set in results_data.results.items():
        if data_set_name in {"AVG", "LM-GAP", "loss"}:
            continue

        for model_name, score in data_set.items():
            if model_name not in model_name_to_scores:
                model_name_to_scores[model_name] = []

            if score is not None:
                model_name_to_scores[model_name].append(score)

    if "AVG" not in results_data.results:
        results_data.results["AVG"] = {}

    for model_name, scores in model_name_to_scores.items():
        if not scores:
            results_data.results["AVG"][model_name] = None
        else:
            score = sum(scores) / len(scores)
            results_data.results["AVG"][model_name] = round_float(score, precision=score_precision)

    return results_data


def reorder_results_with_model_names(
    data: Dict[str, Optional[float]], model_names: List[str]
) -> Dict[str, Optional[float]]:
    ordered_data = {}

    # Keep model names order
    for model_name in model_names:
        if model_name not in data:
            continue

        ordered_data[model_name] = data.pop(model_name)

    # Add rest data at the end
    for model_name in sorted(data.keys()):
        ordered_data[model_name] = data.pop(model_name)

    return ordered_data


def sort_results_with_model_names(data: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    ordered_data = {}

    for model_name in sorted(data.keys()):
        ordered_data[model_name] = data.pop(model_name)

    return ordered_data


def merge_lm_gap_data(
    results_data: AggregatedResults, lm_gap_data_list: List[LMGapResults], score_precision: Optional[int] = None
) -> AggregatedResults:
    if not lm_gap_data_list:
        LOGGER.info("Skipping merging LM-GAP data")
        return results_data

    LOGGER.info("Merging LM-GAP data")
    if "LM-GAP" not in results_data.metrics:
        results_data.metrics["LM-GAP"] = "PerplexityHashed"

    if "LM-GAP" in results_data.results:
        results_lm_gap_data = results_data.results["LM-GAP"]
    else:
        results_lm_gap_data = {}
        results_data.results["LM-GAP"] = results_lm_gap_data

    for lm_gap_data in lm_gap_data_list:
        results_lm_gap_data[lm_gap_data.model_name] = round_float(lm_gap_data.score, precision=score_precision)

    results_data.results["LM-GAP"] = sort_results_with_model_names(results_lm_gap_data)

    return results_data


def merge_loss_data(
    results_data: AggregatedResults, loss_data_list: List[LossStatistics], score_precision: Optional[int] = None
) -> AggregatedResults:
    if not loss_data_list:
        LOGGER.info("Skipping merging loss data")
        return results_data

    LOGGER.info("Merging loss data")
    if "loss" not in results_data.metrics:
        results_data.metrics["loss"] = "CrossEntropy Loss"

    if "loss" in results_data.results:
        results_loss_data = results_data.results["loss"]
    else:
        results_loss_data = {}
        results_data.results["loss"] = results_loss_data

    for loss_data in loss_data_list:
        model_name = loss_data.get_model_name()
        results_loss_data[model_name] = round_float(loss_data.loss, precision=score_precision)

    results_data.results["loss"] = sort_results_with_model_names(results_loss_data)

    return results_data


def generate_aggregated_results_table(
    results_data: AggregatedResults,
    lm_gap_data_list: List[LMGapResults],
    loss_data_list: List[LossStatistics],
    score_precision: Optional[int] = None,
) -> AggregatedResults:
    # Round scores if needed
    if score_precision is not None:
        for dataset_data in results_data.results.values():
            for model_name, model_score in dataset_data.items():
                if model_score is not None:
                    dataset_data[model_name] = round_float(model_score, precision=score_precision)

    results_data = merge_lm_gap_data(results_data, lm_gap_data_list, score_precision=score_precision)
    results_data = merge_loss_data(results_data, loss_data_list, score_precision=score_precision)

    return results_data
