from phd_model_evaluations.data.token_prediction_mlm import TokenPredictionMLM
from phd_model_evaluations.data.tokenized_token import TokenizedToken
from phd_model_evaluations.evaluation.lm_gap.mlm_lm_gap_utils import get_tokens_prediction_mlm


def test_get_tokens_prediction_mlm() -> None:
    tokenized_tokens = [
        TokenizedToken(raw_token="Ġ1990", tokens=["Ġ1990"], indexes=[105]),
        TokenizedToken(raw_token="Ġ1991", tokens=["Ġ19", "91"], indexes=[106, 107]),
        TokenizedToken(raw_token="Ġ1991", tokens=["Ġ19", "92"], indexes=[106, 108]),
        TokenizedToken(raw_token="Ġ1992", tokens=["Ġ19", "8", "9"], indexes=[106, 103, 104]),
        TokenizedToken(raw_token="Ġ1992", tokens=["Ġ19", "8", "8"], indexes=[106, 103, 103]),
        TokenizedToken(raw_token="Ġ1992", tokens=["Ġ19", "8", "7"], indexes=[106, 103, 102]),
        TokenizedToken(raw_token="Ġ1993", tokens=["Ġ19", "93"], indexes=[106, 109]),
        TokenizedToken(raw_token="Ġ2000", tokens=["Ġ2000"], indexes=[200]),
        TokenizedToken(raw_token="Ġ2001", tokens=["Ġ2001"], indexes=[201]),
    ]
    expected_tokens_prediction_mlm = [
        TokenPredictionMLM(
            token_size=1,
            prefix_indexes=[],
            prefix_tokens=[],
            token_indexes=[105, 200, 201],
            token_str_list=["Ġ1990", "Ġ2000", "Ġ2001"],
        ),
        TokenPredictionMLM(
            token_size=2,
            prefix_indexes=[106],
            prefix_tokens=["Ġ19"],
            token_indexes=[107, 108, 109],
            token_str_list=["91", "92", "93"],
        ),
        TokenPredictionMLM(
            token_size=3,
            prefix_indexes=[106, 103],
            prefix_tokens=["Ġ19", "8"],
            token_indexes=[104, 103, 102],
            token_str_list=["9", "8", "7"],
        ),
    ]
    tokens_prediction_mlm = get_tokens_prediction_mlm(tokenized_tokens)
    assert tokens_prediction_mlm == expected_tokens_prediction_mlm, "Invalid tokens for LML prediction"
