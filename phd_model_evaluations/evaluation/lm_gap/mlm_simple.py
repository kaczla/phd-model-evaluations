from typing import List

import torch
from torch.nn import Softmax
from tqdm import tqdm
from transformers import FillMaskPipeline, PreTrainedTokenizer

from phd_model_evaluations.data.lm_gap.lm_gap_prediction_line import LMGapPredictionLine
from phd_model_evaluations.data.prediction_base import PredictionBase
from phd_model_evaluations.data.prediction_for_aggregation import PredictionForAggregation
from phd_model_evaluations.data.prediction_result import PredictionResult
from phd_model_evaluations.data.token_prediction_mlm import TokenPredictionMLM
from phd_model_evaluations.evaluation.lm_gap.mlm_lm_gap_utils import prepare_mlm_data_loader
from phd_model_evaluations.utils.type_utils import MLM_MODEL_TYPES


def run_single_prediction_mlm(
    model_pipeline: FillMaskPipeline,
    tokenized_line: LMGapPredictionLine,
    target_tokens: List[str],
    top_k: int,
) -> List[PredictionResult]:
    assert model_pipeline.tokenizer is not None, "Tokenizer is not initialized!"
    assert model_pipeline.tokenizer.mask_token_id is not None, "Mask token cannot be empty!"
    text_line = model_pipeline.tokenizer.decode(
        tokenized_line.left_context_token_indexes
        + [model_pipeline.tokenizer.mask_token_id]
        + tokenized_line.right_context_token_indexes
    )
    answers = model_pipeline(text_line, targets=target_tokens, top_k=top_k)
    best_answers = answers[:top_k]
    return [
        PredictionResult(
            token=answer["token_str"].strip(),
            token_index=answer["token"],
            score=answer["score"],
        )
        for answer in best_answers
    ]


def run_mlm_simple(
    tokenized_text: List[LMGapPredictionLine],
    tokens_prediction_mlm: List[TokenPredictionMLM],
    model: MLM_MODEL_TYPES,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    batch_size: int,
    top_k: int,
) -> List[PredictionForAggregation]:
    # Get all tokens that the size is equal to 1
    token_prediction_mlm = next(token for token in tokens_prediction_mlm if token.token_size == 1)
    # Use lower top-k if is lower tokens to predict than current top-k
    top_k = len(token_prediction_mlm.token_indexes) if top_k > len(token_prediction_mlm.token_indexes) else top_k
    target_indexes = torch.LongTensor(token_prediction_mlm.token_indexes)

    data_loader = prepare_mlm_data_loader(tokenized_text, tokenizer, batch_size)
    softmax = Softmax(dim=-1)

    all_predictions: List[PredictionForAggregation] = []
    for batch in tqdm(data_loader, total=len(data_loader), desc="Predicting batch of tokens"):
        outputs = model(**batch.to(device))
        # Get indexes of masked tokens - skip first tuple (tensor of row indexes)
        masked_indexes = torch.nonzero(batch.input_ids == tokenizer.mask_token_id, as_tuple=True)[1]
        bach_indexes = torch.arange(start=0, end=outputs.logits.size(0), dtype=torch.long)
        # Get probability of masked tokens
        probabilities = softmax(outputs.logits[bach_indexes, masked_indexes, :])
        # Get probabilities of target tokens
        probabilities = probabilities[..., target_indexes]
        outputs_top_k = torch.topk(probabilities, top_k, largest=True)
        for top_k_indices, top_k_values in zip(
            outputs_top_k.indices.tolist(), outputs_top_k.values.tolist(), strict=True
        ):
            top_k_predictions = []
            for token_indice, token_probability in zip(top_k_indices, top_k_values, strict=True):
                token_index = target_indexes[token_indice].tolist()
                token_text = tokenizer.decode(token_index).strip()
                top_k_predictions.append(PredictionBase(token=token_text, score=token_probability))
            all_predictions.append(PredictionForAggregation(predictions=top_k_predictions, type_score="probability"))

    return all_predictions


def run_mlm_simple_line_by_line(
    tokenized_text: List[LMGapPredictionLine],
    tokens_prediction_mlm: List[TokenPredictionMLM],
    model_pipeline: FillMaskPipeline,
    tokenizer: PreTrainedTokenizer,
    top_k: int,
) -> List[PredictionForAggregation]:
    # Get all tokens that the size is equal to 1
    token_prediction_mlm = next(token for token in tokens_prediction_mlm if token.token_size == 1)

    all_predictions: List[PredictionForAggregation] = []

    for prediction_text in tqdm(tokenized_text, desc="Predicting text", total=len(tokenized_text)):
        top_k_predictions = []
        for result_mlm in run_single_prediction_mlm(
            model_pipeline,
            prediction_text,
            token_prediction_mlm.token_str_list,
            top_k,
        ):
            token = tokenizer.decode(list(token_prediction_mlm.prefix_indexes) + [result_mlm.token_index]).strip()
            top_k_predictions.append(PredictionBase(token=token, score=result_mlm.score))

        all_predictions.append(PredictionForAggregation(predictions=top_k_predictions, type_score="probability"))

    return all_predictions
