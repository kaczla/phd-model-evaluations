import pytest
from transformers import PreTrainedTokenizer

from phd_model_evaluations.evaluation.lm_gap.mlm_lm_gap_utils import get_tokens_for_mlm_lm_gap


def test_get_tokens_for_lm_gap_gpt2(tokenizer_gpt2: PreTrainedTokenizer) -> None:
    lm_gap_tokens = get_tokens_for_mlm_lm_gap(tokenizer_gpt2, 1)
    assert len(lm_gap_tokens) == 1, "Expected token length to be 1"
    assert lm_gap_tokens[0].token_size == 1, "Expected token size to be 1"
    assert len(lm_gap_tokens[0].token_indexes) == 48537, "Invalid number of token indexes"


def test_get_tokens_for_lm_gap_gpt2_smart_tokens(tokenizer_gpt2: PreTrainedTokenizer) -> None:
    lm_gap_tokens = get_tokens_for_mlm_lm_gap(tokenizer_gpt2, 1, smart_tokens=True)
    assert len(lm_gap_tokens) == 1, "Expected token length to be 1"
    assert lm_gap_tokens[0].token_size == 1, "Expected token size to be 1"
    assert len(lm_gap_tokens[0].token_indexes) == 33061, "Invalid number of token indexes"


def test_get_tokens_for_lm_gap_gpt2_length_2(tokenizer_gpt2: PreTrainedTokenizer) -> None:
    with pytest.raises(NotImplementedError):
        get_tokens_for_mlm_lm_gap(tokenizer_gpt2, 2)


def test_get_tokens_for_lm_gap_bert(tokenizer_bert: PreTrainedTokenizer) -> None:
    lm_gap_tokens = get_tokens_for_mlm_lm_gap(tokenizer_bert, 1)
    assert len(lm_gap_tokens) == 1, "Expected token length to be 1"
    assert lm_gap_tokens[0].token_size == 1, "Expected token size to be 1"
    assert len(lm_gap_tokens[0].token_indexes) == 28798, "Invalid number of token indexes"


def test_get_tokens_for_lm_gap_bert_smart_tokens(tokenizer_bert: PreTrainedTokenizer) -> None:
    lm_gap_tokens = get_tokens_for_mlm_lm_gap(tokenizer_bert, 1, smart_tokens=True)
    assert len(lm_gap_tokens) == 1, "Expected token length to be 1"
    assert lm_gap_tokens[0].token_size == 1, "Expected token size to be 1"
    assert len(lm_gap_tokens[0].token_indexes) == 23694, "Invalid number of token indexes"


def test_get_tokens_for_lm_gap_t5(tokenizer_t5: PreTrainedTokenizer) -> None:
    lm_gap_tokens = get_tokens_for_mlm_lm_gap(tokenizer_t5, 1)
    assert len(lm_gap_tokens) == 1, "Expected token length to be 1"
    assert lm_gap_tokens[0].token_size == 1, "Expected token size to be 1"
    assert len(lm_gap_tokens[0].token_indexes) == 31983, "Invalid number of token indexes"


def test_get_tokens_for_lm_gap_t5_smart_tokens(tokenizer_t5: PreTrainedTokenizer) -> None:
    lm_gap_tokens = get_tokens_for_mlm_lm_gap(tokenizer_t5, 1, smart_tokens=True)
    assert len(lm_gap_tokens) == 1, "Expected token length to be 1"
    assert lm_gap_tokens[0].token_size == 1, "Expected token size to be 1"
    assert len(lm_gap_tokens[0].token_indexes) == 21984, "Invalid number of token indexes"
