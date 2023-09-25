from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import FillMaskPipeline, PreTrainedTokenizer

from phd_model_evaluations.data.lm_gap.lm_gap_prediction_line import LMGapPredictionLine
from phd_model_evaluations.data.prediction_for_aggregation import PredictionForAggregation
from phd_model_evaluations.data.prediction_result import PredictionResult
from phd_model_evaluations.data.token_prediction_mlm import TokenPredictionMLM
from phd_model_evaluations.evaluation.lm_gap.mlm_loss import run_single_prediction_loss
from phd_model_evaluations.evaluation.lm_gap.mlm_simple import run_single_prediction_mlm
from phd_model_evaluations.utils.prediction_utils import convert_from_loss_to_probabilities
from phd_model_evaluations.utils.type_utils import MLM_MODEL_TYPES


def run_mlm_mixed(
    tokenized_text: List[LMGapPredictionLine],
    tokens_prediction_mlm: List[TokenPredictionMLM],
    model_pipeline: FillMaskPipeline,
    model: MLM_MODEL_TYPES,
    tokenizer: PreTrainedTokenizer,
    torch_device: torch.device,
    batch_size: int,
    top_k: int,
) -> List[PredictionForAggregation]:
    all_predictions: List[PredictionForAggregation] = []

    for prediction_text in tqdm(tokenized_text, desc="Predicting text", total=len(tokenized_text)):
        # Run MLM prediction
        mlm_prediction_results: List[Tuple[TokenPredictionMLM, PredictionResult]] = []
        for token_prediction_mlm in tqdm(tokens_prediction_mlm, desc="Predicting tokens with MLM", leave=False):
            for result_mlm in run_single_prediction_mlm(
                model_pipeline,
                prediction_text,
                token_prediction_mlm.token_str_list,
                top_k,
            ):
                mlm_prediction_results.append((token_prediction_mlm, result_mlm))
        tokens_mlm: List[List[int]] = [
            list(token_prediction_mlm.prefix_indexes) + [result_mlm.token_index]
            for token_prediction_mlm, result_mlm in mlm_prediction_results
        ]

        # Run loss prediction base on loss
        predicted_tokens = run_single_prediction_loss(
            model=model,
            tokenizer=tokenizer,
            tokenized_line=prediction_text,
            token_labels_list=tokens_mlm,
            device=torch_device,
            batch_size=batch_size,
            best_n=top_k * len(tokens_prediction_mlm),
        )
        predicted_tokens_with_probabilities = convert_from_loss_to_probabilities(predicted_tokens)

        all_predictions.append(predicted_tokens_with_probabilities)

    return all_predictions
