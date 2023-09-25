import pytest

from phd_model_evaluations.utils.tokenization_utils import clean_token


@pytest.mark.parametrize(
    ("token", "expected_token"),
    [
        ("", ""),
        ("\t", ""),
        ("\n\t\n", ""),
        ("\ntoken\n", "token"),
        ("token\t", "token"),
        ("\ttoken", "token"),
        ("token", "token"),
        ("Abc", "Abc"),
        ("aBc", "aBc"),
    ],
)
def test_clean_token(token: str, expected_token: str) -> None:
    result = clean_token(token)
    assert result == expected_token, "Invalid cleaned token"
