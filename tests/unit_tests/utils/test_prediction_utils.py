from phd_model_evaluations.data.prediction_base import PredictionBase
from phd_model_evaluations.data.prediction_for_aggregation import PredictionForAggregation
from phd_model_evaluations.evaluation.lm_gap.lm_gap_utils import group_aggregated_prediction
from phd_model_evaluations.utils.prediction_utils import (
    fix_predictions,
    fix_predictions_with_unk_probability,
    softmax_predictions,
)


def test_softmax_predictions_single_1() -> None:
    results = softmax_predictions([PredictionBase(token="my_token", score=1.0)], top_k=5)
    assert len(results) == 1, "Invalid number of results"
    assert results[0].token == "my_token", "Invalid token name"
    assert results[0].score == 1.0, "Invalid token score"


def test_softmax_predictions_single_2() -> None:
    results = softmax_predictions([PredictionBase(token="my_token", score=0.83)], top_k=5)
    assert len(results) == 1, "Invalid number of results"
    assert results[0].token == "my_token", "Invalid token name"
    assert results[0].score == 1.0, "Invalid token score"


def test_softmax_predictions_many_1() -> None:
    results = softmax_predictions(
        [
            PredictionBase(token=".", score=0.4),
            PredictionBase(token=",", score=0.3),
            PredictionBase(token="!", score=0.2),
        ],
        top_k=5,
    )
    assert len(results) == 3, "Invalid number of results"
    assert [result.token for result in results] == [".", ",", "!"], "Invalid token names"
    assert [result.score for result in results] == [
        0.36716540111092544,
        0.3322249935333472,
        0.3006096053557273,
    ], "Invalid token scores"


def test_softmax_predictions_many_2() -> None:
    results = softmax_predictions([PredictionBase(token=str(i), score=0.1) for i in range(10)], top_k=5)
    assert len(results) == 5, "Invalid number of results"
    assert [result.token for result in results] == ["0", "1", "2", "3", "4"], "Invalid token names"
    assert [result.score for result in results] == [0.2, 0.2, 0.2, 0.2, 0.2], "Invalid token scores"


def test_softmax_predictions_many_3() -> None:
    results = softmax_predictions(
        [
            PredictionBase(token=".", score=16),
            PredictionBase(token=",", score=10),
            PredictionBase(token="!", score=8),
        ],
        top_k=2,
    )
    assert len(results) == 2, "Invalid number of results"
    assert [result.token for result in results] == [".", ","], "Invalid token names"
    assert [result.score for result in results] == [0.9975273768433651, 0.0024726231566347743], "Invalid token scores"


def test_softmax_predictions_many_4() -> None:
    results = softmax_predictions([PredictionBase(token=str(i), score=1 / (i + 2)) for i in range(10)], top_k=5)
    assert len(results) == 5, "Invalid number of results"
    assert [result.token for result in results] == ["0", "1", "2", "3", "4"], "Invalid token names"
    assert [result.score for result in results] == [
        0.2449400265304485,
        0.20733725615224682,
        0.19075948446744395,
        0.18145603462801962,
        0.17550719822184097,
    ], "Invalid token scores"


def test_fix_predictions_single_1() -> None:
    results = fix_predictions([PredictionBase(token="my_token", score=1.0)], top_k=5)
    assert len(results) == 1, "Invalid number of results"
    assert results[0].token == "my_token", "Invalid token name"
    assert results[0].score == 1.0, "Invalid token score"


def test_fix_predictions_single_2() -> None:
    results = fix_predictions([PredictionBase(token="my_token", score=0.83)], top_k=5)
    assert len(results) == 1, "Invalid number of results"
    assert results[0].token == "my_token", "Invalid token name"
    assert results[0].score == 1.0, "Invalid token score"


def test_fix_predictions_many_1() -> None:
    results = fix_predictions(
        [
            PredictionBase(token=".", score=0.4),
            PredictionBase(token=",", score=0.3),
            PredictionBase(token="!", score=0.2),
        ],
        top_k=5,
    )
    assert len(results) == 3, "Invalid number of results"
    assert [result.token for result in results] == [".", ",", "!"], "Invalid token names"
    assert [result.score for result in results] == [
        0.44444444444444453,
        0.33333333333333337,
        0.22222222222222227,
    ], "Invalid token scores"


def test_fix_predictions_many_2() -> None:
    results = fix_predictions([PredictionBase(token=str(i), score=0.1) for i in range(10)], top_k=5)
    assert len(results) == 5, "Invalid number of results"
    assert [result.token for result in results] == ["0", "1", "2", "3", "4"], "Invalid token names"
    assert [result.score for result in results] == [0.2, 0.2, 0.2, 0.2, 0.2], "Invalid token scores"


def test_fix_predictions_many_3() -> None:
    results = fix_predictions(
        [
            PredictionBase(token=".", score=16),
            PredictionBase(token=",", score=10),
            PredictionBase(token="!", score=8),
        ],
        top_k=2,
    )
    assert len(results) == 2, "Invalid number of results"
    assert [result.token for result in results] == [".", ","], "Invalid token names"
    assert [result.score for result in results] == [0.6153846153846154, 0.38461538461538464], "Invalid token scores"


def test_fix_predictions_many_4() -> None:
    results = fix_predictions([PredictionBase(token=str(i), score=1 / (i + 2)) for i in range(10)], top_k=5)
    assert len(results) == 5, "Invalid number of results"
    assert [result.token for result in results] == ["0", "1", "2", "3", "4"], "Invalid token names"
    assert [result.score for result in results] == [
        0.3448275862068966,
        0.22988505747126436,
        0.1724137931034483,
        0.13793103448275862,
        0.11494252873563218,
    ], "Invalid token scores"


def test_fix_predictions_with_unk_probability_many_() -> None:
    results = fix_predictions_with_unk_probability(
        [
            PredictionBase(token=".", score=0.4),
            PredictionBase(token=",", score=0.3),
            PredictionBase(token="!", score=0.2),
        ],
        top_k=5,
        unk_probability=0.15,
    )
    assert len(results) == 4, "Invalid number of results"
    assert [result.token for result in results] == [".", ",", "!", ""], "Invalid token names"
    assert [result.score for result in results] == [
        0.37777777777777793,
        0.28333333333333344,
        0.18888888888888897,
        0.15,
    ], "Invalid token scores"


def test_fix_predictions_with_unk_probability_many_2() -> None:
    results = fix_predictions_with_unk_probability(
        [
            PredictionBase(token=".", score=0.4),
            PredictionBase(token=",", score=0.3),
            PredictionBase(token="!", score=0.2),
        ],
        top_k=2,
        unk_probability=0.15,
    )
    assert len(results) == 3, "Invalid number of results"
    assert [result.token for result in results] == [".", ",", ""], "Invalid token names"
    assert [result.score for result in results] == [
        0.48571428571428577,
        0.3642857142857143,
        0.15,
    ], "Invalid token scores"


def test_group_aggregated_prediction_fiter_out_empty() -> None:
    aggregated_prediction = PredictionForAggregation(
        type_score="probability",
        predictions=[
            PredictionBase(token="This", score=0.25),
            PredictionBase(token="", score=0.2),
            PredictionBase(token="I", score=0.128),
        ],
    )
    predictions = group_aggregated_prediction(aggregated_prediction)
    assert len(predictions) == 2, "Invalid number for grouped prediction"
    assert predictions == [
        PredictionBase(token="This", score=0.25),
        PredictionBase(token="I", score=0.128),
    ], "Invalid grouped prediction"


def test_group_aggregated_prediction_aggregate() -> None:
    aggregated_prediction = PredictionForAggregation(
        type_score="probability",
        predictions=[
            PredictionBase(token="This", score=0.3),
            PredictionBase(token="I", score=0.25),
            PredictionBase(token="This", score=0.2),
            PredictionBase(token="this", score=0.128),
            PredictionBase(token="This", score=0.02),
        ],
    )
    predictions = group_aggregated_prediction(aggregated_prediction)
    assert len(predictions) == 3, "Invalid number for grouped prediction"
    assert predictions == [
        PredictionBase(token="This", score=0.52),
        PredictionBase(token="I", score=0.25),
        PredictionBase(token="this", score=0.128),
    ], "Invalid grouped prediction"


def test_group_aggregated_prediction() -> None:
    aggregated_prediction = PredictionForAggregation(
        type_score="probability",
        predictions=[
            PredictionBase(token="This", score=0.3),
            PredictionBase(token="I", score=0.25),
            PredictionBase(token="This", score=0.2),
            PredictionBase(token="", score=0.02),
        ],
    )
    predictions = group_aggregated_prediction(aggregated_prediction)
    assert len(predictions) == 2, "Invalid number for grouped prediction"
    assert predictions == [
        PredictionBase(token="This", score=0.5),
        PredictionBase(token="I", score=0.25),
    ], "Invalid grouped prediction"
