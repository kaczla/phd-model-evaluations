#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, AutoConfig, HfArgumentParser

from phd_model_evaluations.cli.evaluation.lm_gap.evaluate_seq2seq_model_arguments import LMGapEvaluateSeq2SeqArguments
from phd_model_evaluations.cli.evaluation.lm_gap.evaluate_utils import get_file_out_path
from phd_model_evaluations.cli.evaluation.lm_gap.lm_gap_arguments import LMGapEvaluateArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.arguments.model_arguments import ModelArguments
from phd_model_evaluations.data.lm_gap.lm_gap_context_line import LMGapContextLine
from phd_model_evaluations.evaluation.lm_gap.clm_lm_gap_utils import convert_prediction_tokens_to_aggregation
from phd_model_evaluations.evaluation.lm_gap.lm_gap_utils import (
    load_lm_gap_context_lines,
    prepare_prediction_lm_gap_text,
    save_lm_gap_predictions,
)
from phd_model_evaluations.evaluation.lm_gap.seq2seq_lm_gap_utils import (
    get_seq2seq_predict_token_index,
    tokenize_seq2seq_lm_gap_text,
)
from phd_model_evaluations.evaluation.lm_gap.seq2seq_simple import run_seq2seq_simple
from phd_model_evaluations.utils.model_utils import (
    check_padding_token,
    get_seq2seq_model,
    get_sequence_length,
    get_tokenizer,
)

LOGGER = logging.getLogger(__name__)


def parse_args(
    cmd_args: Optional[List[str]] = None,
) -> Tuple[LMGapEvaluateArguments, LMGapEvaluateSeq2SeqArguments, ModelArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (LMGapEvaluateArguments, LMGapEvaluateSeq2SeqArguments, ModelArguments, LoggerArguments),
        description="Run LM-GAP prediction for seq2seq (encoder-decoder) models.",
    )
    lm_gap_evaluation, seq2seq_evaluation, model_args, logger_args = parser.parse_args_into_dataclasses(
        args=cmd_args, look_for_args_file=False
    )
    return lm_gap_evaluation, seq2seq_evaluation, model_args, logger_args


def run_seq2seq(
    lm_gap_text: List[LMGapContextLine],
    file_out_path: Path,
    model_arguments: ModelArguments,
    lm_gap_args: LMGapEvaluateArguments,
    method_arguments: LMGapEvaluateSeq2SeqArguments,
) -> None:
    LOGGER.info(f"Using model: {model_arguments.model_name}")
    # Load tokenizer nad seq2seq model
    tokenizer = get_tokenizer(model_arguments)
    model = get_seq2seq_model(model_arguments)
    check_padding_token(tokenizer, model)
    model.eval()
    # Tokenize text
    sequence_length = (
        get_sequence_length(model_arguments.sequence_length, tokenizer) - tokenizer.num_special_tokens_to_add()
    )
    predict_token, predict_token_index = get_seq2seq_predict_token_index(tokenizer)
    tokenized_text = tokenize_seq2seq_lm_gap_text(tokenizer, lm_gap_text, predict_token)
    tokenized_text = prepare_prediction_lm_gap_text(tokenized_text, 1, sequence_length)

    all_prediction_tokens = run_seq2seq_simple(
        tokenized_text,
        model,
        tokenizer,
        predict_token_index,
        model_arguments.batch_size,
        lm_gap_args.top_k,
        method_arguments,
    )
    all_predictions = convert_prediction_tokens_to_aggregation(tokenizer, all_prediction_tokens)

    save_lm_gap_predictions(
        file_out_path,
        all_predictions,
        lm_gap_args.top_k,
        unk_probability=lm_gap_args.unk_probability,
        save_aggregations=lm_gap_args.save_aggregations,
    )


def main() -> None:
    lm_gap_args, method_args, model_args, logger_args = parse_args()
    set_logging_from_logger_arguments(logger_args)

    model_name = model_args.model_name
    model_type = AutoConfig.from_pretrained(model_name).model_type
    assert MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING is not None, "Cannot check seq2seq models"
    all_seq2seq_models = {config.model_type for config in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys()}
    if model_type not in all_seq2seq_models:
        LOGGER.error(
            f'Not found "{model_name}" (type: {model_type}) as seq2seq model! Models for seq2seq: {all_seq2seq_models}'
        )
        raise ValueError(f'Not found "{model_name}" (type: {model_type}) as seq2seq model!')

    file_out_path = get_file_out_path(model_args, lm_gap_args, method_args)
    lm_gap_text = load_lm_gap_context_lines(
        lm_gap_args.file_in,
        lm_gap_args.column_left_context,
        lm_gap_args.column_right_context,
    )
    LOGGER.info(f"Output will be saved in: {file_out_path}")

    run_seq2seq(lm_gap_text, file_out_path, model_args, lm_gap_args, method_args)


if __name__ == "__main__":
    main()
