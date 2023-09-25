from pathlib import Path

import pytest
from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer


@pytest.fixture(scope="session")
def tokenizer_bert(path_tokenizer_bert: Path) -> BertTokenizer:
    return BertTokenizer.from_pretrained(path_tokenizer_bert)


@pytest.fixture(scope="session")
def tokenizer_gpt2(path_tokenizer_gpt2: Path) -> GPT2Tokenizer:
    return GPT2Tokenizer.from_pretrained(path_tokenizer_gpt2)


@pytest.fixture(scope="session")
def tokenizer_t5(path_tokenizer_t5: Path) -> T5Tokenizer:
    return T5Tokenizer.from_pretrained(path_tokenizer_t5)
