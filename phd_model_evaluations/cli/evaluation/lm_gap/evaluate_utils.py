import logging
from pathlib import Path
from typing import List, Union

from phd_model_evaluations.cli.evaluation.lm_gap.evaluate_clm_model_arguments import LMGapEvaluateCLMArguments
from phd_model_evaluations.cli.evaluation.lm_gap.evaluate_mlm_model_arguments import LMGapEvaluateMLMArguments
from phd_model_evaluations.cli.evaluation.lm_gap.evaluate_seq2seq_model_arguments import LMGapEvaluateSeq2SeqArguments
from phd_model_evaluations.cli.evaluation.lm_gap.lm_gap_arguments import LMGapEvaluateArguments
from phd_model_evaluations.data.arguments.model_arguments import ModelArguments

LOGGER = logging.getLogger(__name__)


def get_tags_from_arguments(
    model_arguments: ModelArguments,
    evaluation_arguments: LMGapEvaluateArguments,
    method_arguments: Union[LMGapEvaluateMLMArguments, LMGapEvaluateCLMArguments, LMGapEvaluateSeq2SeqArguments],
) -> List[str]:
    model_name = model_arguments.get_model_name()
    tags = [f"model_name={model_name}", f"top_k={evaluation_arguments.top_k}"]

    if isinstance(method_arguments, LMGapEvaluateMLMArguments):
        tags.append(f"method={method_arguments.method}")
        tags.append(f"token_length={method_arguments.max_token_size}")

    elif isinstance(method_arguments, (LMGapEvaluateCLMArguments, LMGapEvaluateSeq2SeqArguments)):
        tags.append(f"depth={method_arguments.depth}")

    else:
        raise RuntimeError(f"Unsupported method arguments type: {type(method_arguments)}")

    if evaluation_arguments.smart_tokens:
        tags.append("smart_token=True")

    return tags


def get_file_out_path(
    model_arguments: ModelArguments,
    evaluation_arguments: LMGapEvaluateArguments,
    method_arguments: Union[LMGapEvaluateMLMArguments, LMGapEvaluateCLMArguments, LMGapEvaluateSeq2SeqArguments],
) -> Path:
    out_path = evaluation_arguments.file_out
    if evaluation_arguments.generate_out_file_name:
        directory_path = out_path if out_path.is_dir() else out_path.parent
        file_tags = get_tags_from_arguments(model_arguments, evaluation_arguments, method_arguments)
        file_name = f"out-{','.join(file_tags)}.tsv"
        file_out_path = directory_path / file_name
        if evaluation_arguments.overwrite_output and file_out_path.exists():
            LOGGER.info(f"Overwriting output: {file_out_path}")
            return file_out_path

        assert not file_out_path.exists(), f"Output file: {file_out_path} exists!"
        return file_out_path

    assert not out_path.is_dir(), f"Output path should not be a directory: {out_path}"
    if evaluation_arguments.overwrite_output and out_path.exists():
        LOGGER.info(f"Overwriting output: {out_path}")
    elif not evaluation_arguments.overwrite_output:
        assert not out_path.exists(), f"Output file {out_path} exists!"

    return out_path
