#!/usr/bin/env python3

"""
Example run:
CUDA_VISIBLE_DEVICES='0' TRANSFORMERS_CACHE='.cache/transformers' \
python -m phd_model_evaluations.cli.evaluation.model.compute_loss \
  --model_name roberta-base \
  --file_path dev-0/raw_data.txt.xz
"""

import logging
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.evaluation.model.compute_loss_arguments import ComputeLossArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.arguments.model_arguments import ModelArguments
from phd_model_evaluations.evaluation.model.compute_loss import compute_loss
from phd_model_evaluations.evaluation.model.loss_utils import save_loss_statistics
from phd_model_evaluations.utils.model_utils import get_model, get_tokenizer

LOGGER = logging.getLogger(__name__)


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[ComputeLossArguments, ModelArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (ComputeLossArguments, ModelArguments, LoggerArguments),
        description="Compute loss of given model and text file."
        " It will save loss statistics in `statistics-loss.json` file (in the same directory as input file)"
        " if save path missing. If save file path exists, then new statistics will be appended this file.",
    )
    compute_loss_args, model_args, logger_args = parser.parse_args_into_dataclasses(
        args=cmd_args, look_for_args_file=False
    )
    return compute_loss_args, model_args, logger_args


def main() -> None:
    compute_loss_args, model_args, logger_args = parse_args()
    set_logging_from_logger_arguments(logger_args)

    file_path = compute_loss_args.file_path

    tokenizer = get_tokenizer(model_args)
    max_length = tokenizer.model_max_length if model_args.sequence_length is None else model_args.sequence_length

    model = get_model(model_args)
    model.eval()

    loss_statistics = compute_loss(
        file_path,
        model,
        tokenizer,
        model_args.batch_size,
        max_length,
        model.device,
        model_human_name=model_args.model_human_name,
        pad_to_max_length=model_args.pad_to_max_length,
        join_examples=compute_loss_args.join_examples,
    )
    loss_statistics.model_name = model_args.model_name

    save_path = (
        file_path.parent / "statistics-loss.json"
        if compute_loss_args.save_path is None
        else compute_loss_args.save_path
    )
    LOGGER.info(f"Computed loss: {loss_statistics.loss}")
    save_loss_statistics(save_path, loss_statistics)


if __name__ == "__main__":
    main()
