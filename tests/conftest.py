from pathlib import Path

import pytest

from phd_model_evaluations.data.tokenization.sub_word_type_detector import SubWordTypeDetector


@pytest.fixture(scope="session")
def path_tests() -> Path:
    return Path(__file__).parent


@pytest.fixture(scope="session")
def path_resources(path_tests: Path) -> Path:
    return path_tests / "resources"


@pytest.fixture(scope="session")
def path_models(path_resources: Path) -> Path:
    return path_resources / "models"


@pytest.fixture(scope="session")
def path_roberta_base_model(path_models: Path) -> Path:
    return path_models / "roberta-base"


@pytest.fixture(scope="session")
def path_gpt2_base_model(path_models: Path) -> Path:
    return path_models / "gpt2-base"


@pytest.fixture(scope="session")
def path_t5_small_model(path_models: Path) -> Path:
    return path_models / "t5-small"


@pytest.fixture(scope="session")
def path_t5_base_model(path_models: Path) -> Path:
    return path_models / "t5-base"


@pytest.fixture(scope="session")
def path_tokenizer(path_resources: Path) -> Path:
    return path_resources / "tokenizer"


@pytest.fixture(scope="session")
def path_tokenizer_bert(path_tokenizer: Path) -> Path:
    return path_tokenizer / "bert"


@pytest.fixture(scope="session")
def path_tokenizer_gpt2(path_tokenizer: Path) -> Path:
    return path_tokenizer / "gpt2"


@pytest.fixture(scope="session")
def path_tokenizer_t5(path_tokenizer: Path) -> Path:
    return path_tokenizer / "t5"


@pytest.fixture(scope="session")
def sub_word_type_detector_gpt2() -> SubWordTypeDetector:
    return SubWordTypeDetector(whole_word_begging="Ġ")


@pytest.fixture(scope="session")
def sub_word_type_detector_bert() -> SubWordTypeDetector:
    return SubWordTypeDetector(inner_word_begging="##")


@pytest.fixture(scope="session")
def sub_word_type_detector_t5() -> SubWordTypeDetector:
    return SubWordTypeDetector(whole_word_begging="▁")
