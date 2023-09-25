#!/usr/bin/env python3

"""
Example run:
CUDA_VISIBLE_DEVICES='0' TRANSFORMERS_CACHE='.cache/transformers' \
python -m phd_model_evaluations.cli.evaluation.lm_gap.evaluate_mlm_model \
  --model_name roberta-base \
  --file_in dev-0/in.tsv.xz \
  --file_out dev-0/out.tsv \
  --method simple
"""


import logging
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import MODEL_FOR_MASKED_LM_MAPPING, AutoConfig, FillMaskPipeline, HfArgumentParser, pipeline

from phd_model_evaluations.cli.evaluation.lm_gap.evaluate_mlm_model_arguments import LMGapEvaluateMLMArguments
from phd_model_evaluations.cli.evaluation.lm_gap.evaluate_utils import get_file_out_path
from phd_model_evaluations.cli.evaluation.lm_gap.lm_gap_arguments import LMGapEvaluateArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.arguments.model_arguments import ModelArguments
from phd_model_evaluations.data.lm_gap.lm_gap_context_line import LMGapContextLine
from phd_model_evaluations.data.lm_gap.prediction_type_mlm import PredictionTypeMLM
from phd_model_evaluations.evaluation.lm_gap.lm_gap_utils import (
    load_lm_gap_context_lines,
    prepare_prediction_lm_gap_text,
    save_lm_gap_predictions,
)
from phd_model_evaluations.evaluation.lm_gap.mlm_lm_gap_utils import get_tokens_for_mlm_lm_gap, tokenize_mlm_lm_gap_text
from phd_model_evaluations.evaluation.lm_gap.mlm_loss import run_mlm_loss
from phd_model_evaluations.evaluation.lm_gap.mlm_mixed import run_mlm_mixed
from phd_model_evaluations.evaluation.lm_gap.mlm_simple import run_mlm_simple
from phd_model_evaluations.utils.model_utils import get_mlm_model, get_sequence_length, get_tokenizer

LOGGER = logging.getLogger(__name__)


def parse_args(
    cmd_args: Optional[List[str]] = None,
) -> Tuple[LMGapEvaluateArguments, LMGapEvaluateMLMArguments, ModelArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (LMGapEvaluateArguments, LMGapEvaluateMLMArguments, ModelArguments, LoggerArguments),
        description="Run LM-GAP prediction for MLM (encoder) models.",
    )
    lm_gap_evaluation, mlm_evaluation, model_args, logger_args = parser.parse_args_into_dataclasses(
        args=cmd_args, look_for_args_file=False
    )
    return lm_gap_evaluation, mlm_evaluation, model_args, logger_args


def run_mlm(
    lm_gap_text: List[LMGapContextLine],
    file_out_path: Path,
    model_arguments: ModelArguments,
    lm_gap_args: LMGapEvaluateArguments,
    method_args: LMGapEvaluateMLMArguments,
) -> None:
    LOGGER.info(f"Using model: {model_arguments.model_name}")

    # Load tokenizer and get tokens
    tokenizer = get_tokenizer(model_arguments)
    tokens_prediction_mlm = get_tokens_for_mlm_lm_gap(
        tokenizer, method_args.max_token_size, smart_tokens=lm_gap_args.smart_tokens
    )
    max_token_size_from_tokens = max(
        [token_prediction_mlm.token_size for token_prediction_mlm in tokens_prediction_mlm]
    )

    # Tokenize text
    sequence_length = (
        get_sequence_length(model_arguments.sequence_length, tokenizer) - tokenizer.num_special_tokens_to_add()
    )
    tokenized_text = tokenize_mlm_lm_gap_text(tokenizer, lm_gap_text)
    tokenized_text = prepare_prediction_lm_gap_text(tokenized_text, max_token_size_from_tokens, sequence_length)

    # Load MLM model
    model = get_mlm_model(model_arguments)
    model.eval()
    model_pipeline = pipeline(
        task="fill-mask",
        model=model,
        tokenizer=tokenizer,
        top_k=lm_gap_args.top_k,
        batch_size=1,
        device=model.device,
    )
    assert isinstance(
        model_pipeline, FillMaskPipeline
    ), f"Expecting MLM model (FillMaskPipeline pipeline, got: {model_pipeline.__class__.__name__})"

    method = method_args.method
    if method == PredictionTypeMLM.simple:
        all_predictions = run_mlm_simple(
            tokenized_text,
            tokens_prediction_mlm,
            model,
            tokenizer,
            model.device,
            model_arguments.batch_size,
            lm_gap_args.top_k,
        )
    elif method == PredictionTypeMLM.loss:
        all_predictions = run_mlm_loss(
            tokenized_text,
            tokens_prediction_mlm,
            model,
            tokenizer,
            model.device,
            model_arguments.batch_size,
            lm_gap_args.top_k,
        )
    elif method == PredictionTypeMLM.mixed:
        all_predictions = run_mlm_mixed(
            tokenized_text,
            tokens_prediction_mlm,
            model_pipeline,
            model,
            tokenizer,
            model.device,
            model_arguments.batch_size,
            lm_gap_args.top_k,
        )
    else:
        raise RuntimeError(f"Unsupported method: {method} for prediction MLM")

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
    assert MODEL_FOR_MASKED_LM_MAPPING is not None, "Cannot check MLM models"
    all_mlm_models = {config.model_type for config in MODEL_FOR_MASKED_LM_MAPPING.keys()}
    if model_type not in all_mlm_models:
        LOGGER.error(f'Not found "{model_name}" (type: {model_type}) as MLM model! Models for MLM: {all_mlm_models}')
        raise ValueError(f'Not found "{model_name}" (type: {model_type}) as MLM model!')

    file_out_path = get_file_out_path(model_args, lm_gap_args, method_args)
    lm_gap_text = load_lm_gap_context_lines(
        lm_gap_args.file_in,
        lm_gap_args.column_left_context,
        lm_gap_args.column_right_context,
    )
    LOGGER.info(f"Output will be saved in: {file_out_path}")

    run_mlm(lm_gap_text, file_out_path, model_args, lm_gap_args, method_args)


if __name__ == "__main__":
    main()
