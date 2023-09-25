#!/usr/bin/env python3


import logging
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, AutoConfig, HfArgumentParser

from phd_model_evaluations.cli.evaluation.lm_gap.evaluate_clm_model_arguments import LMGapEvaluateCLMArguments
from phd_model_evaluations.cli.evaluation.lm_gap.evaluate_utils import get_file_out_path
from phd_model_evaluations.cli.evaluation.lm_gap.lm_gap_arguments import LMGapEvaluateArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.arguments.model_arguments import ModelArguments
from phd_model_evaluations.data.lm_gap.lm_gap_context_line import LMGapContextLine
from phd_model_evaluations.evaluation.lm_gap.clm_lm_gap_utils import (
    convert_prediction_tokens_to_aggregation,
    tokenize_clm_lm_gap_text,
)
from phd_model_evaluations.evaluation.lm_gap.clm_simple import run_clm_simple
from phd_model_evaluations.evaluation.lm_gap.lm_gap_utils import (
    load_lm_gap_context_lines,
    prepare_prediction_lm_gap_text,
    save_lm_gap_predictions,
)
from phd_model_evaluations.utils.model_utils import (
    check_padding_token,
    get_clm_model,
    get_sequence_length,
    get_tokenizer,
)

LOGGER = logging.getLogger(__name__)


def parse_args(
    cmd_args: Optional[List[str]] = None,
) -> Tuple[LMGapEvaluateArguments, LMGapEvaluateCLMArguments, ModelArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (LMGapEvaluateArguments, LMGapEvaluateCLMArguments, ModelArguments, LoggerArguments),
        description="Run LM-GAP prediction for CLM (decoder) models.",
    )
    lm_gap_evaluation, clm_evaluation, model_args, logger_args = parser.parse_args_into_dataclasses(
        args=cmd_args, look_for_args_file=False
    )
    return lm_gap_evaluation, clm_evaluation, model_args, logger_args


def run_clm(
    lm_gap_text: List[LMGapContextLine],
    file_out_path: Path,
    model_arguments: ModelArguments,
    lm_gap_args: LMGapEvaluateArguments,
    method_arguments: LMGapEvaluateCLMArguments,
) -> None:
    LOGGER.info(f"Using model: {model_arguments.model_name}")

    # Load tokenizer nad CLM model
    tokenizer = get_tokenizer(model_arguments)
    model = get_clm_model(model_arguments)
    check_padding_token(tokenizer, model)
    model.eval()

    # Tokenize text
    sequence_length = (
        get_sequence_length(model_arguments.sequence_length, tokenizer) - tokenizer.num_special_tokens_to_add()
    )
    tokenized_text = tokenize_clm_lm_gap_text(tokenizer, lm_gap_text)
    tokenized_text = prepare_prediction_lm_gap_text(tokenized_text, method_arguments.depth, sequence_length)

    all_prediction_tokens = run_clm_simple(
        tokenized_text,
        model,
        tokenizer,
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
    assert MODEL_FOR_CAUSAL_LM_MAPPING is not None, "Cannot check CLM models"
    all_clm_models = {config.model_type for config in MODEL_FOR_CAUSAL_LM_MAPPING.keys()}
    if model_type not in all_clm_models:
        LOGGER.error(f'Not found "{model_name}" (type: {model_type}) as CLM model! Models for CLM: {all_clm_models}')
        raise ValueError(f'Not found "{model_name}" (type: {model_type}) as CLM model!')

    file_out_path = get_file_out_path(model_args, lm_gap_args, method_args)
    lm_gap_text = load_lm_gap_context_lines(
        lm_gap_args.file_in,
        lm_gap_args.column_left_context,
        lm_gap_args.column_right_context,
    )
    LOGGER.info(f"Output will be saved in: {file_out_path}")

    run_clm(lm_gap_text, file_out_path, model_args, lm_gap_args, method_args)


if __name__ == "__main__":
    main()
