#!/usr/bin/env python3

import logging
from typing import List, Optional, Tuple

from transformers import AutoConfig, HfArgumentParser

from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.arguments.model_arguments import ModelArguments
from phd_model_evaluations.utils.model_utils import get_model, get_tokenizer

LOGGER = logging.getLogger(__name__)


def parse_args(
    cmd_args: Optional[List[str]] = None,
) -> Tuple[ModelArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (ModelArguments, LoggerArguments),
        description="Check model contains all weights required for LM-GAP and fine-tuning.",
    )
    model_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)

    return model_args, logger_args


def check_model() -> None:
    model_args, logger_args = parse_args()
    set_logging_from_logger_arguments(logger_args)

    LOGGER.info(f"Checking model from: {model_args.model_name}")

    config = AutoConfig.from_pretrained(model_args.model_name)
    LOGGER.info(f"Loaded configuration: {config.__class__.__name__}")

    tokenizer = get_tokenizer(model_args)
    LOGGER.info(f"Loaded tokenizer: {tokenizer.__class__.__name__}")

    model = get_model(model_args)
    LOGGER.info(f"Loaded model: {model.__class__.__name__}")

    LOGGER.info("Everything OK")


if __name__ == "__main__":
    check_model()
