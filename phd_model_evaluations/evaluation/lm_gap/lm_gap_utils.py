import json
import logging
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from tqdm.contrib import tzip

from phd_model_evaluations.data.lm_gap.lm_gap_context_line import LMGapContextLine
from phd_model_evaluations.data.lm_gap.lm_gap_context_line_with_gap import LMGapContextLineWithGap
from phd_model_evaluations.data.lm_gap.lm_gap_prediction_line import LMGapPredictionLine
from phd_model_evaluations.data.prediction_base import PredictionBase
from phd_model_evaluations.data.prediction_for_aggregation import PredictionForAggregation
from phd_model_evaluations.utils.common_utils import get_open_fn
from phd_model_evaluations.utils.prediction_utils import fix_predictions_with_unk_probability

LOGGER = logging.getLogger(__name__)


def load_lm_gap_context_lines_with_gap(
    input_file_path: Path, expected_file_path: Path, column_left_context: int, column_right_context: int
) -> List[LMGapContextLineWithGap]:
    LOGGER.info(f"Loading LM-GAP text from file: {input_file_path} and gaps from: {expected_file_path}")
    open_fn_input_file_path = get_open_fn(input_file_path.name)
    open_fn_expected_file_path = get_open_fn(expected_file_path.name)

    text_with_gap: List[LMGapContextLineWithGap] = []
    with (
        open_fn_input_file_path(input_file_path, "rt") as f_input,
        open_fn_expected_file_path(expected_file_path, "rt") as f_expected,
    ):
        for input_line, expected_line in tzip(f_input, f_expected, desc="Loading text and gaps"):
            input_line = input_line.rstrip("\r\n")
            split_input_line = input_line.split("\t")
            left_context = split_input_line[column_left_context].strip()
            right_context = split_input_line[column_right_context].strip()
            gap = expected_line.strip()
            text_with_gap.append(
                LMGapContextLineWithGap(left_context=left_context, right_context=right_context, gap=gap)
            )

    LOGGER.info(f"Loaded {len(text_with_gap)} LM-GAP text")
    return text_with_gap


def load_lm_gap_context_lines(
    file_path: Path, column_left_context: int, column_right_context: int
) -> List[LMGapContextLine]:
    LOGGER.info(f"Loading LM-GAP text from file: {file_path}")
    text = []
    open_fn = get_open_fn(file_path.name)
    with open_fn(file_path, "rt") as f_read:
        for line in tqdm(f_read, desc="Loading lines"):
            line = line.rstrip("\r\n")
            split_line = line.split("\t")
            left_context = split_line[column_left_context].strip()
            right_context = split_line[column_right_context].strip()
            text.append(LMGapContextLine(left_context=left_context, right_context=right_context))
    LOGGER.info(f"Loaded {len(text):,d} lines")
    return text


def group_aggregated_prediction(aggregated_prediction: PredictionForAggregation) -> List[PredictionBase]:
    # Filter out empty tokens after tokenization
    predictions = [prediction for prediction in aggregated_prediction.predictions if prediction.token]

    # Group duplicated tokens after tokenization
    token_to_list_id: Dict[str, int] = {}
    ids_to_remove = []
    for i, prediction in enumerate(predictions):
        token_id = token_to_list_id.get(prediction.token)
        if token_id is None:
            token_to_list_id[prediction.token] = i

        else:
            predictions[token_id].score += prediction.score
            ids_to_remove.append(i)

    # Remove merged tokens and resort predictions
    if ids_to_remove:
        for id_to_remove in reversed(ids_to_remove):
            del predictions[id_to_remove]

        return sorted(predictions, key=lambda x: x.score, reverse=True)

    return predictions


def save_lm_gap_predictions(
    save_file_path: Path,
    all_predictions: List[PredictionForAggregation],
    top_k: int,
    unk_probability: float = 0.001,
    save_aggregations: bool = False,
) -> None:
    LOGGER.info(f"Saving predictions into: {save_file_path}")
    open_fn = get_open_fn(save_file_path.name)
    with open_fn(save_file_path, "wt") as f_write:
        for predictions in tqdm(all_predictions, desc="Saving predictions"):
            predictions_with_unk = fix_predictions_with_unk_probability(
                group_aggregated_prediction(predictions),
                top_k,
                unk_probability=unk_probability,
            )
            f_write.write(" ".join([f"{prediction.token}:{prediction.score}" for prediction in predictions_with_unk]))
            f_write.write("\n")
    LOGGER.info(f"Saved {len(all_predictions):,d} predictions")

    if save_aggregations:
        save_lm_gap_aggregation(save_file_path, all_predictions)


def save_lm_gap_aggregation(save_file_path: Path, all_predictions: List[PredictionForAggregation]) -> None:
    open_fn = get_open_fn(save_file_path.name)
    all_data_file_path = save_file_path.parent / ("aggregation_" + save_file_path.stem + ".jsonl")
    LOGGER.info(f"Saving all prediction data into: {all_data_file_path}")
    with open_fn(all_data_file_path, "wt") as f_write:
        for i_predictions in tqdm(all_predictions, desc="Saving aggregation"):
            f_write.write(f"{json.dumps(i_predictions.dict(), indent=None, ensure_ascii=False)}\n")
    LOGGER.info("Aggregation saved")


def prepare_prediction_lm_gap_text(
    tokenized_text: List[LMGapPredictionLine],
    max_token_length: int,
    max_sentence_length: int,
    min_context_length: int = 15,
) -> List[LMGapPredictionLine]:
    LOGGER.info(f"Preparing prediction text for tokens length: {max_token_length}")
    results = []
    for i, tokenized_line in enumerate(
        tqdm(
            tokenized_text,
            desc="Prepare predictions text",
        )
    ):
        line_id = i + 1
        tokens_before_length = len(tokenized_line.left_context_token_indexes)
        tokens_after_length = len(tokenized_line.right_context_token_indexes)
        line_length = tokens_before_length + tokens_after_length + max_token_length

        if line_length <= max_sentence_length:
            results.append(tokenized_line)
            continue

        else:
            # context before and context after + MASK token/s
            min_sentence_length = min_context_length * 2 + max_token_length
            assert (
                min_sentence_length <= max_sentence_length
            ), f"[{line_id}] Context length too long ({min_context_length})"

            line_length_with_before_context_with_mask_token = tokens_before_length + max_token_length
            line_length_with_after_context = max_sentence_length - line_length_with_before_context_with_mask_token
            # Trim after context
            if line_length_with_after_context > 0 and line_length_with_after_context > min_context_length:
                tokens_before = tokenized_line.left_context_token_indexes
                tokens_after = tokenized_line.right_context_token_indexes[:line_length_with_after_context]
                line_length = len(tokens_before) + len(tokens_after) + max_token_length
                LOGGER.debug(
                    f"[{line_id}] Trim after context from {tokens_after_length} to"
                    f" {len(tokens_after)} tokens, before context length: {len(tokens_before)}"
                    f" max sentence length: {max_sentence_length} and final line length: {line_length}"
                )

            # Trim before and after context
            else:
                line_length_half = (max_sentence_length - max_token_length) // 2

                tokens_before = tokenized_line.left_context_token_indexes[-line_length_half:]
                tokens_after = tokenized_line.right_context_token_indexes[:line_length_half]
                if len(tokens_before) < len(tokens_after):
                    tokens_filled_length = line_length_half - len(tokens_before)
                    tokens_after = tokenized_line.right_context_token_indexes[
                        : (line_length_half + tokens_filled_length)
                    ]
                elif len(tokens_after) < len(tokens_before):
                    tokens_filled_length = line_length_half - len(tokens_after)
                    tokens_before = tokenized_line.left_context_token_indexes[
                        -(line_length_half + tokens_filled_length) :
                    ]

                line_length = len(tokens_before) + len(tokens_after) + max_token_length
                LOGGER.debug(
                    f"[{line_id}] Trim before context from {tokens_before_length} to"
                    f" {len(tokens_before)} tokens  and after context from {tokens_after_length} to"
                    f" {len(tokens_after)} tokens, where half length is: {line_length_half},"
                    f" context length: {min_context_length}, max sentence length: {max_sentence_length}"
                    f" and final line length: {line_length}"
                )

            results.append(
                LMGapPredictionLine(
                    left_context_token_indexes=tokens_before,
                    right_context_token_indexes=tokens_after,
                )
            )

    LOGGER.info(f"Prepared {len(results):,d} lines")
    return results
