from typing import List

from phd_model_evaluations.data.input_with_prediction_tokens import InputWithPredictionTokens


class InputWithSeq2SeqPredictionTokens(InputWithPredictionTokens):
    decoder_input_indexes: List[int]
    prediction_token_index: int
