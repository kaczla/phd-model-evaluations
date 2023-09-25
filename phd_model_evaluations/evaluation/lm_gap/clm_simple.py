from typing import List

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from phd_model_evaluations.cli.evaluation.lm_gap.evaluate_clm_model_arguments import LMGapEvaluateCLMArguments
from phd_model_evaluations.data.input_with_prediction_tokens import InputWithPredictionTokens
from phd_model_evaluations.data.lm_gap.lm_gap_prediction_line import LMGapPredictionLine
from phd_model_evaluations.data.prediction_token import PredictionToken
from phd_model_evaluations.evaluation.lm_gap.clm_lm_gap_utils import (
    prepare_clm_data_loader,
    prepare_clm_data_loader_for_next_prediction,
)
from phd_model_evaluations.utils.type_utils import CLM_MODEL_TYPES


def process_next_predictions(
    model: CLM_MODEL_TYPES,
    tokenizer: PreTrainedTokenizer,
    input_with_predicted_tokens: InputWithPredictionTokens,
    batch_size: int,
    current_depth: int,
    max_depth: int,
    top_k: int,
    top_k_reduce: int,
    top_k_minimum: int,
) -> List[PredictionToken]:
    tokenizer_special_indexes = set(tokenizer.all_special_ids)
    dataloader = prepare_clm_data_loader_for_next_prediction(tokenizer, input_with_predicted_tokens, batch_size)

    predicted_tokens = []
    for batch in dataloader:
        index_predicted_tokens = batch.pop("index_predicted_tokens").tolist()
        outputs: CausalLMOutputWithCrossAttentions = model(**batch.to(model.device))
        next_tokens_scores = torch.softmax(outputs.logits[:, -1, :], dim=-1)
        top_k_tokens = torch.topk(next_tokens_scores, top_k, largest=True)
        # Join each line with top-k predictions
        for index, (token_index_list, probability_list) in zip(
            index_predicted_tokens,
            zip(top_k_tokens.indices.tolist(), top_k_tokens.values.tolist(), strict=True),
            strict=True,
        ):
            previous_predicted_token = input_with_predicted_tokens.predicted_tokens[index]
            # Get predicted tokens
            predicted_tokens.extend(
                [
                    previous_predicted_token.create_next_prediction_token(token_index, probability)
                    for token_index, probability in zip(token_index_list, probability_list, strict=True)
                    if token_index not in tokenizer_special_indexes
                ]
            )

    current_depth += 1
    # Do deeper prediction
    if current_depth < max_depth:
        # Update scores for next predictions
        for i in range(len(input_with_predicted_tokens.predicted_tokens)):
            input_with_predicted_tokens.predicted_tokens[i].total_score *= 0.5
            input_with_predicted_tokens.predicted_tokens[i].scores.append(0.5)

        next_top_k = max(top_k_minimum, top_k - top_k_reduce)
        predicted_tokens.extend(
            process_next_predictions(
                model,
                tokenizer,
                InputWithPredictionTokens(
                    input_indexes=input_with_predicted_tokens.input_indexes, predicted_tokens=predicted_tokens
                ),
                batch_size,
                current_depth,
                max_depth,
                next_top_k,
                top_k_reduce,
                top_k_minimum,
            )
        )
    # Return final predictions
    return predicted_tokens


def run_clm_simple(
    tokenized_text: List[LMGapPredictionLine],
    model: CLM_MODEL_TYPES,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    top_k: int,
    method_arguments: LMGapEvaluateCLMArguments,
) -> List[List[PredictionToken]]:
    tokenizer_special_indexes = set(tokenizer.all_special_ids)
    data_loader = prepare_clm_data_loader(tokenized_text, tokenizer)

    all_prediction_tokens: List[List[PredictionToken]] = []
    for batch in tqdm(data_loader, total=len(data_loader), desc="Predicting batch of tokens"):
        outputs: CausalLMOutputWithCrossAttentions = model(**batch.to(model.device))
        next_tokens_scores = torch.softmax(outputs.logits[:, -1, :], dim=-1)
        top_k_tokens = torch.topk(next_tokens_scores, top_k, largest=True)
        # Join each line with top-k predictions
        for prefix_input_ids, (token_index_list, probability_list) in zip(
            batch["input_ids"].tolist(),
            zip(top_k_tokens.indices.tolist(), top_k_tokens.values.tolist(), strict=True),
            strict=True,
        ):
            # Get predictions
            predicted_tokens = [
                PredictionToken(token_indexes=[token_index], scores=[probability], total_score=probability)
                for token_index, probability in zip(token_index_list, probability_list, strict=True)
                if token_index not in tokenizer_special_indexes
            ]
            # Create predicted tokens with corresponding input
            input_with_predicted_tokens = InputWithPredictionTokens(
                input_indexes=prefix_input_ids, predicted_tokens=predicted_tokens
            )

            if method_arguments.depth > 1:
                # Update scores for next predictions
                for i in range(len(input_with_predicted_tokens.predicted_tokens)):
                    input_with_predicted_tokens.predicted_tokens[i].total_score *= 0.5
                    input_with_predicted_tokens.predicted_tokens[i].scores.append(0.5)

                next_top_k = max(method_arguments.top_k_minimum, top_k - method_arguments.top_k_reduce)
                predicted_tokens.extend(
                    process_next_predictions(
                        model,
                        tokenizer,
                        input_with_predicted_tokens,
                        batch_size,
                        1,
                        method_arguments.depth,
                        next_top_k,
                        method_arguments.top_k_reduce,
                        method_arguments.top_k_minimum,
                    )
                )

            all_prediction_tokens.append(predicted_tokens)

    return all_prediction_tokens
