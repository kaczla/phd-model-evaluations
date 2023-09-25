from pathlib import Path

from transformers import GPT2Tokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from phd_model_evaluations.data.arguments.model_arguments import ModelArguments
from phd_model_evaluations.utils.model_utils import get_sequence_length, get_tokenizer


def test_get_tokenizer(path_tokenizer_gpt2: Path) -> None:
    model_arguments = ModelArguments(model_name=str(path_tokenizer_gpt2), use_fast_tokenizer=False)
    tokenizer = get_tokenizer(model_arguments)
    assert isinstance(tokenizer, PreTrainedTokenizer), "Expected pretrained tokenizer type"
    assert not isinstance(tokenizer, PreTrainedTokenizerFast), "Should not be fast version of tokenizer"


def test_get_tokenizer_fast(path_tokenizer_gpt2: Path) -> None:
    model_arguments = ModelArguments(model_name=str(path_tokenizer_gpt2), use_fast_tokenizer=True)
    tokenizer = get_tokenizer(model_arguments)
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Expected fast pretrained tokenizer type"


def test_get_sequence_length(tokenizer_gpt2: GPT2Tokenizer) -> None:
    sequence_length = get_sequence_length(None, tokenizer_gpt2)
    assert sequence_length == 1024, "Invalid sequence length for GPT-2 tokenizer"


def test_get_sequence_length_2(tokenizer_gpt2: GPT2Tokenizer) -> None:
    sequence_length = get_sequence_length(348, tokenizer_gpt2)
    assert sequence_length == 348, "Invalid sequence length for GPT-2 tokenizer"


def test_get_sequence_length_3(tokenizer_gpt2: GPT2Tokenizer) -> None:
    sequence_length = get_sequence_length(5000, tokenizer_gpt2)
    assert sequence_length == 1024, "Invalid sequence length for GPT-2 tokenizer"
