from math import ceil
from typing import List

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from phd_model_evaluations.data.lm_gap.lm_gap_prediction_line import LMGapPredictionLine
from phd_model_evaluations.data.prediction_base import PredictionBase
from phd_model_evaluations.data.prediction_for_aggregation import PredictionForAggregation
from phd_model_evaluations.data.token_prediction_mlm import TokenPredictionMLM
from phd_model_evaluations.utils.common_utils import batchify
from phd_model_evaluations.utils.prediction_utils import convert_from_loss_to_probabilities
from phd_model_evaluations.utils.type_utils import MLM_MODEL_TYPES


def run_single_prediction_loss(
    model: MLM_MODEL_TYPES,
    tokenizer: PreTrainedTokenizer,
    tokenized_line: LMGapPredictionLine,
    token_labels_list: List[List[int]],
    device: torch.device,
    batch_size: int,
    best_n: int = 1,
) -> PredictionForAggregation:
    assert tokenizer.mask_token_id is not None, "Mask token cannot be empty!"
    mask_token_id = tokenizer.mask_token_id
    top_k_predictions = []
    best_result = float("inf")
    best_token = ""

    all_inputs: List[List[int]] = []
    all_labels: List[List[int]] = []
    for token_labels in token_labels_list:
        all_inputs.append(
            tokenized_line.left_context_token_indexes
            + [mask_token_id] * len(token_labels)
            + tokenized_line.right_context_token_indexes
        )
        all_labels.append(
            tokenized_line.left_context_token_indexes + token_labels + tokenized_line.right_context_token_indexes
        )

    total_batches = ceil(len(all_inputs) / batch_size)
    batches_inputs = batchify(all_inputs, batch_size)
    batches_labels = batchify(all_labels, batch_size)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for batch_i, (batch_inputs, batch_labels) in enumerate(
        tqdm(
            zip(batches_inputs, batches_labels, strict=True),
            total=total_batches,
            desc="Predicting batch of tokens",
            leave=False,
        )
    ):
        batch_tokens_start_index = batch_i * batch_size
        input_data = tokenizer(
            tokenizer.batch_decode(batch_inputs, padding=True),
            padding=True,
            return_tensors="pt",
        ).to(device)
        labels = tokenizer(tokenizer.batch_decode(batch_labels, padding=True), padding=True, return_tensors="pt")[
            "input_ids"
        ].to(device)
        outputs = model(**input_data, labels=labels)
        loss_items = loss_fct(outputs.logits.permute(0, 2, 1), labels).mean(dim=1)
        loss_topk = torch.topk(loss_items, len(batch_inputs), largest=False)
        loss_topk_values = loss_topk.values.tolist()
        loss_topk_indices = loss_topk.indices.tolist()

        loss = loss_topk_values[0]
        if loss < best_result:
            best_result = loss
            tokens_indices = loss_topk_indices[0] + batch_tokens_start_index
            best_token = tokenizer.decode(token_labels_list[tokens_indices]).strip()

        # Add all predictions
        if best_n > 1:
            for token_indice, loss in zip(loss_topk_indices, loss_topk_values, strict=True):
                tokens_indices = token_indice + batch_tokens_start_index
                token_text = tokenizer.decode(token_labels_list[tokens_indices]).strip()
                top_k_predictions.append(PredictionBase(token=token_text, score=loss))

            if len(top_k_predictions) > (2 * best_n):
                top_k_predictions = sorted(top_k_predictions, key=lambda x: x.score)[:best_n]

    top_k_predictions = (
        sorted(top_k_predictions, key=lambda x: x.score)[:best_n]
        if best_n > 1
        else [PredictionBase(token=best_token, score=best_result)]
    )
    return PredictionForAggregation(predictions=top_k_predictions, type_score="loss")


def run_mlm_loss(
    tokenized_text: List[LMGapPredictionLine],
    tokens_prediction_mlm: List[TokenPredictionMLM],
    model: MLM_MODEL_TYPES,
    tokenizer: PreTrainedTokenizer,
    torch_device: torch.device,
    batch_size: int,
    top_k: int,
) -> List[PredictionForAggregation]:
    all_predictions: List[PredictionForAggregation] = []
    tokens_mlm = [
        token_prediction_mlm.prefix_indexes + [token_index]
        for token_prediction_mlm in tokens_prediction_mlm
        for token_index in token_prediction_mlm.token_indexes
    ]

    for prediction_text in tqdm(tokenized_text, desc="Predicting text", total=len(tokenized_text)):
        predicted_tokens = run_single_prediction_loss(
            model=model,
            tokenizer=tokenizer,
            tokenized_line=prediction_text,
            token_labels_list=tokens_mlm,
            device=torch_device,
            batch_size=batch_size,
            best_n=top_k,
        )
        predicted_tokens_with_probabilities = convert_from_loss_to_probabilities(predicted_tokens)

        all_predictions.append(predicted_tokens_with_probabilities)

    return all_predictions
