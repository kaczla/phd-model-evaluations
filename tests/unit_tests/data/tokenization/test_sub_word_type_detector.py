import pytest

from phd_model_evaluations.data.tokenization.sub_word_type_detector import SubWordTypeDetector

# Inner tokens #


@pytest.mark.parametrize("token", ["##tier", "##ing"])
def test_is_inner_word_bert(sub_word_type_detector_bert: SubWordTypeDetector, token: str) -> None:
    assert sub_word_type_detector_bert.is_inner_word(token), "It should be inner token"


@pytest.mark.parametrize("token", ["ing", "ware"])
def test_is_inner_word_gpt2(sub_word_type_detector_gpt2: SubWordTypeDetector, token: str) -> None:
    assert sub_word_type_detector_gpt2.is_inner_word(token), "It should be inner token"


@pytest.mark.parametrize("token", ["scribe", "ives"])
def test_is_inner_word_t5(sub_word_type_detector_t5: SubWordTypeDetector, token: str) -> None:
    assert sub_word_type_detector_t5.is_inner_word(token), "It should be inner token"


@pytest.mark.parametrize("token", ["buy", "This"])
def test_is_not_inner_word_bert(sub_word_type_detector_bert: SubWordTypeDetector, token: str) -> None:
    assert not sub_word_type_detector_bert.is_inner_word(token), "It should not be inner token"


@pytest.mark.parametrize("token", ["Ġsell", "ĠFound"])
def test_is_not_inner_word_gpt2(sub_word_type_detector_gpt2: SubWordTypeDetector, token: str) -> None:
    assert not sub_word_type_detector_gpt2.is_inner_word(token), "It should not be inner token"


@pytest.mark.parametrize("token", ["▁height", "▁Perhaps"])
def test_is_not_inner_word_t5(sub_word_type_detector_t5: SubWordTypeDetector, token: str) -> None:
    assert not sub_word_type_detector_t5.is_inner_word(token), "It should not be inner token"


# Whole tokens #


@pytest.mark.parametrize("token", ["buy", "This"])
def test_is_whole_word_bert(sub_word_type_detector_bert: SubWordTypeDetector, token: str) -> None:
    assert sub_word_type_detector_bert.is_whole_word(token), "It should be whole token"


@pytest.mark.parametrize("token", ["Ġsell", "ĠFound"])
def test_is_whole_word_gpt2(sub_word_type_detector_gpt2: SubWordTypeDetector, token: str) -> None:
    assert sub_word_type_detector_gpt2.is_whole_word(token), "It should be whole token"


@pytest.mark.parametrize("token", ["▁height", "▁Perhaps"])
def test_is_whole_word_t5(sub_word_type_detector_t5: SubWordTypeDetector, token: str) -> None:
    assert sub_word_type_detector_t5.is_whole_word(token), "It should be whole token"


@pytest.mark.parametrize("token", ["##tier", "##ing"])
def test_is_not_whole_word_bert(sub_word_type_detector_bert: SubWordTypeDetector, token: str) -> None:
    assert not sub_word_type_detector_bert.is_whole_word(token), "It should not be whole token"


@pytest.mark.parametrize("token", ["ing", "ware"])
def test_is_not_whole_word_gpt2(sub_word_type_detector_gpt2: SubWordTypeDetector, token: str) -> None:
    assert not sub_word_type_detector_gpt2.is_whole_word(token), "It should not be whole token"


@pytest.mark.parametrize("token", ["scribe", "ives"])
def test_is_not_whole_word_t5(sub_word_type_detector_t5: SubWordTypeDetector, token: str) -> None:
    assert not sub_word_type_detector_t5.is_whole_word(token), "It should not be whole token"
