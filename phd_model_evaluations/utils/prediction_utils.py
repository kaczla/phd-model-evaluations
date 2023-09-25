import math
from typing import List

from phd_model_evaluations.data.prediction_base import PredictionBase
from phd_model_evaluations.data.prediction_for_aggregation import PredictionForAggregation


def convert_from_loss_to_probabilities(
    prediction_for_aggregation: PredictionForAggregation,
) -> PredictionForAggregation:
    prediction_scores = [math.log(prediction.score, 0.5) for prediction in prediction_for_aggregation.predictions]
    sum_scores = sum(prediction_scores)
    return PredictionForAggregation(
        type_score="probability",
        predictions=[
            PredictionBase(token=prediction.token, score=prediction_score / sum_scores)
            for prediction, prediction_score in zip(
                prediction_for_aggregation.predictions, prediction_scores, strict=True
            )
        ],
    )


def fix_predictions(predictions: List[PredictionBase], top_k: int) -> List[PredictionBase]:
    if len(predictions) == top_k and sum([prediction.score for prediction in predictions]) == 1.0:
        return predictions

    top_k_predictions = predictions[:top_k]
    sum_score = sum([prediction.score for prediction in top_k_predictions])

    return [
        PredictionBase(token=prediction.token, score=prediction.score / sum_score) for prediction in top_k_predictions
    ]


def fix_predictions_with_unk_probability(
    predictions: List[PredictionBase], top_k: int, unk_probability: float = 0.0001
) -> List[PredictionBase]:
    top_k_predictions = fix_predictions(predictions, top_k)
    sum_probabilities = sum([prediction.score for prediction in top_k_predictions])
    sum_probabilities -= unk_probability
    return [
        PredictionBase(token=prediction.token, score=prediction.score * sum_probabilities)
        for prediction in top_k_predictions
    ] + [PredictionBase(token="", score=unk_probability)]


def softmax_predictions(predictions: List[PredictionBase], top_k: int) -> List[PredictionBase]:
    if len(predictions) == top_k and sum([prediction.score for prediction in predictions]) == 1.0:
        return predictions

    top_k_predictions = predictions[:top_k]
    exp_list = [math.exp(prediction.score) for prediction in top_k_predictions]
    sum_exp = sum(exp_list)

    return [
        PredictionBase(token=prediction.token, score=exp_score / sum_exp)
        for prediction, exp_score in zip(top_k_predictions, exp_list, strict=True)
    ]
