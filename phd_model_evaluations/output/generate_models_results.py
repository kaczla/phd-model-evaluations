import json
import logging
from itertools import product
from pathlib import Path
from typing import Dict, List, Union

from phd_model_evaluations.data.results.aggregated_results import AggregatedResults
from phd_model_evaluations.data.results.dataset_result import DatasetResult
from phd_model_evaluations.data.results.empty_dataset_result import EmptyDatasetResult
from phd_model_evaluations.data.results.evaluation_result import EvaluationResult
from phd_model_evaluations.utils.dataset_utils import DATASET_NAMES_TO_BEST_METRIC

LOGGER = logging.getLogger(__name__)

MAP_METRIC_NAME_TO_HUMAN_NAME: Dict[str, str] = {
    "accuracy": "Accuracy",
    "matthews_correlation": "Matthews correlation",
    "f1": "F1 score",
    "spearmanr": "Spearman correlation",
}


MAP_DATASET_NAME_TO_HUMAN_NAME: Dict[str, str] = {
    "cola": "CoLA",
    "mnli": "MNLI",
    "mnli-m": "MNLI-m",
    "mnli-mm": "MNLI-mm",
    "mrpc": "MRPC",
    "qnli": "QNLI",
    "qqp": "QQP",
    "rte": "RTE",
    "sst2": "SST-2",
    "stsb": "STS-B",
    "wnli": "WNLI",
}

MAP_DATASET_NAME_WITH_SET_NAME_TO_DATASET_NAME: Dict[str, Dict[str, str]] = {
    "mnli": {
        "matched": "mnli-m",
        "mismatched": "mnli-mm",
    }
}

DATASET_NAMES = [
    "cola",
    "mnli-m",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
]


def get_human_metric_name(dataset_name: str) -> str:
    name = DATASET_NAMES_TO_BEST_METRIC[("glue", dataset_name)]
    return MAP_METRIC_NAME_TO_HUMAN_NAME[name]


def get_human_dataset_name(dataset_name: str) -> str:
    return MAP_DATASET_NAME_TO_HUMAN_NAME[dataset_name]


def get_evaluation_results(search_path: Path, model_name: str, dataset_name: str) -> List[EvaluationResult]:
    path = search_path / f"{model_name}-{dataset_name}"

    if not path.is_dir():
        LOGGER.error(
            f'Not found model directory for model: "{model_name}" and dataset: "{dataset_name}" in path: {search_path}'
        )
        return []

    evaluation_file_path = path / "evaluation_output.json"
    if not evaluation_file_path.is_file():
        LOGGER.error(
            f'Not found evaluation file for model: "{model_name}" and dataset: "{dataset_name}" in path: {path}'
        )
        return []

    evaluation_data_list: List[Dict] = json.loads(evaluation_file_path.read_text())
    if not evaluation_data_list or "best_metric" not in evaluation_data_list[0]:
        LOGGER.error(
            f'Cannot get evaluation data for model: "{model_name}" and dataset: "{dataset_name}" in path: {path}'
        )
        return []

    return [EvaluationResult(**evaluation_data) for evaluation_data in evaluation_data_list]


def get_evaluation_results_for_set_names(
    evaluation_results: List[EvaluationResult], set_name_to_dataset_name: Dict[str, str], model_name: str
) -> List[Union[DatasetResult, EmptyDatasetResult]]:
    dataset_results: List[Union[DatasetResult, EmptyDatasetResult]] = []

    for set_name, dataset_name in set_name_to_dataset_name.items():
        for evaluation_result in evaluation_results:
            if set_name in evaluation_result.set_name:
                dataset_results.append(DatasetResult(dataset_name=dataset_name, score=evaluation_result.best_metric))
                break
        else:
            dataset_results.append(EmptyDatasetResult(dataset_name=dataset_name))
            LOGGER.error(
                f'Not found evaluation results for set name: "{set_name}", model: "{model_name}"'
                f' and dataset: "{dataset_name}"'
            )

    return dataset_results


def get_dataset_results(
    search_path: Path, model_name: str, dataset_name: str
) -> List[Union[DatasetResult, EmptyDatasetResult]]:
    evaluation_results = get_evaluation_results(search_path, model_name, dataset_name)

    # Get evaluation results for each split name
    if dataset_name in MAP_DATASET_NAME_WITH_SET_NAME_TO_DATASET_NAME:
        return get_evaluation_results_for_set_names(
            evaluation_results, MAP_DATASET_NAME_WITH_SET_NAME_TO_DATASET_NAME[dataset_name], model_name
        )

    # Not found evaluation results
    elif not evaluation_results:
        return [EmptyDatasetResult(dataset_name=dataset_name)]

    elif len(evaluation_results) > 1:
        raise RuntimeError(f"Found more than 1 evaluation records, found: {len(evaluation_results)} evaluation records")

    return [DatasetResult(dataset_name=dataset_name, score=evaluation_results[0].best_metric)]


def generate_models_results(
    search_path: Path,
    model_names: List[str],
    search_dataset_names: List[str],
    return_empty_score: bool = False,
    score_factor: float = 100.0,
) -> AggregatedResults:
    models_results = AggregatedResults(
        model_list=model_names,
        metrics={
            get_human_dataset_name(dataset_name): get_human_metric_name(dataset_name) for dataset_name in DATASET_NAMES
        },
        results={get_human_dataset_name(dataset_name): {} for dataset_name in DATASET_NAMES},
    )
    for model_name, dataset_name in product(model_names, search_dataset_names):
        LOGGER.debug(f'Searching evaluation results for model: "{model_name}" and dataset: "{dataset_name}"')

        dataset_results = get_dataset_results(search_path, model_name, dataset_name)
        for dataset_result in dataset_results:
            human_dataset_name = get_human_dataset_name(dataset_result.dataset_name)

            score = dataset_result.get_score(score_factor=score_factor)
            if score is None and not return_empty_score:
                continue

            # Change 0.0 score to None
            if score is not None and score == 0.0:
                LOGGER.warning(
                    f'Detect 0.0 score, setting `None` value for model: "{model_name}"'
                    f' and dataset: "{dataset_result.dataset_name}"'
                )
                score = None

            LOGGER.info(f'Found score: {score} for model: "{model_name}" and dataset: "{dataset_result.dataset_name}"')
            models_results.results[human_dataset_name][model_name] = score

    return models_results
