from typing import List

import pytest

from phd_model_evaluations.data.results.aggregated_results import AggregatedResults
from phd_model_evaluations.data.results.lm_gap_results import LMGapResults
from phd_model_evaluations.data.statistics.loss_statistics import LossStatistics
from phd_model_evaluations.output.generate_aggregated_results import (
    compute_average_score,
    merge_lm_gap_data,
    merge_loss_data,
)


@pytest.fixture()
def aggregated_results() -> AggregatedResults:
    return AggregatedResults(
        model_list=["RoBERTa-base", "BERT-base", "RoBERTa-large"],
        metrics={"CoLA": "Matthews correlation", "QQP": "F1 score", "QNLI": "Accuracy", "RTE": "Accuracy"},
        results={
            "CoLA": {"BERT-base": 60.2, "RoBERTa-base": 63.6, "RoBERTa-large": 68.0},
            "QNLI": {"BERT-base": 91.3, "RoBERTa-base": 92.8, "RoBERTa-large": 94.7},
            "QQP": {"BERT-base": 91.3, "RoBERTa-base": 91.9, "RoBERTa-large": 92.2},
            "RTE": {"BERT-base": 77.7, "RoBERTa-base": 78.7, "RoBERTa-large": 86.6},
        },
    )


@pytest.fixture()
def lm_gap_data_list() -> List[LMGapResults]:
    return [
        LMGapResults(model_name="BERT-base", score=756.6),
        LMGapResults(model_name="RoBERTa-base", score=114.2),
        LMGapResults(model_name="RoBERTa-large", score=86.1),
    ]


@pytest.fixture()
def loss_data_list() -> List[LossStatistics]:
    return [
        LossStatistics(
            model_name="BERT-base", model_class_name="BERT", join_examples=True, sequence_length=512, loss=2.365
        ),
        LossStatistics(
            model_name="roberta-base",
            model_class_name="RoBERTa",
            model_human_name="RoBERTa-base",
            join_examples=True,
            sequence_length=512,
            loss=0.8981,
        ),
        LossStatistics(
            model_name="roberta-large",
            model_class_name="RoBERTa",
            model_human_name="RoBERTa-large",
            join_examples=True,
            sequence_length=512,
            loss=0.263,
        ),
    ]


@pytest.fixture()
def aggregated_results_with_lm_gap(aggregated_results: AggregatedResults) -> AggregatedResults:
    aggregated_results_copy = aggregated_results.copy(deep=True)
    aggregated_results_copy.results["LM-GAP"] = {"BERT-base": 756.6, "RoBERTa-base": 114.2, "RoBERTa-large": 86.1}
    aggregated_results_copy.metrics["LM-GAP"] = "PerplexityHashed"
    return aggregated_results_copy


@pytest.fixture()
def aggregated_results_with_loss(aggregated_results: AggregatedResults) -> AggregatedResults:
    aggregated_results_copy = aggregated_results.copy(deep=True)
    aggregated_results_copy.results["loss"] = {"RoBERTa-base": 0.8981, "RoBERTa-large": 0.263, "BERT-base": 2.365}
    aggregated_results_copy.metrics["loss"] = "CrossEntropy Loss"
    return aggregated_results_copy


@pytest.fixture()
def aggregated_results_with_average_score(aggregated_results_with_lm_gap: AggregatedResults) -> AggregatedResults:
    aggregated_results_copy = aggregated_results_with_lm_gap.copy(deep=True)
    aggregated_results_copy.results["AVG"] = {"RoBERTa-base": 81.75, "RoBERTa-large": 85.375, "BERT-base": 80.125}
    aggregated_results_copy.metrics["AVG"] = "Average score"
    return aggregated_results_copy


def test_compute_average_score(
    aggregated_results_with_lm_gap: AggregatedResults, aggregated_results_with_average_score: AggregatedResults
) -> None:
    assert "AVG" not in aggregated_results_with_lm_gap.results, '"AVG" data should no be exist in input data'
    result = compute_average_score(aggregated_results_with_lm_gap)
    assert result == aggregated_results_with_average_score, "Invalid aggregated results after computing average score"


def test_merge_lm_gap_data(
    aggregated_results: AggregatedResults,
    lm_gap_data_list: List[LMGapResults],
    aggregated_results_with_lm_gap: AggregatedResults,
) -> None:
    assert "LM-GAP" not in aggregated_results.results, '"LM-GAP" data should not be exist in input data'
    result = merge_lm_gap_data(aggregated_results, lm_gap_data_list)
    assert "LM-GAP" in result.results, '"LM-GAP" data should be in aggregated results'
    assert result == aggregated_results_with_lm_gap, "Invalid aggregated results after merging LM-GAP data"


def test_merge_loss_data(
    aggregated_results: AggregatedResults,
    loss_data_list: List[LossStatistics],
    aggregated_results_with_loss: AggregatedResults,
) -> None:
    assert "loss" not in aggregated_results.results, '"loss" data should not be exist in input data'
    result = merge_loss_data(aggregated_results, loss_data_list)
    assert "loss" in result.results, '"loss" data should be in aggregated results'
    assert result == aggregated_results_with_loss, "Invalid aggregated results after merging loss data"
