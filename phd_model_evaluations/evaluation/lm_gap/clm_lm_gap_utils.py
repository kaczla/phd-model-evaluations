import logging
from collections import Counter
from typing import Counter as CounterType
from typing import List

from datasets.arrow_dataset import Dataset
from datasets.formatting.formatting import LazyBatch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedTokenizer

from phd_model_evaluations.data.input_with_prediction_tokens import InputWithPredictionTokens
from phd_model_evaluations.data.lm_gap.lm_gap_context_line import LMGapContextLine
from phd_model_evaluations.data.lm_gap.lm_gap_prediction_line import LMGapPredictionLine
from phd_model_evaluations.data.prediction_base import PredictionBase
from phd_model_evaluations.data.prediction_for_aggregation import PredictionForAggregation
from phd_model_evaluations.data.prediction_token import PredictionToken
from phd_model_evaluations.utils.tokenization_utils import clean_token

LOGGER = logging.getLogger(__name__)


def convert_prediction_tokens(
    tokenizer: PreTrainedTokenizer, prediction_tokens: List[PredictionToken]
) -> List[PredictionBase]:
    predictions = []
    token_to_number_token_indexes = {}
    for prediction_token in prediction_tokens:
        token = tokenizer.decode(prediction_token.token_indexes)
        token_clean = clean_token(token)
        if not token_clean or " " in token_clean:
            continue

        # Check token contains correct indexes (without invalid indexes) - check indexes are the same after decoding
        token_indexes = tokenizer.encode(token)
        if token_indexes == prediction_token.token_indexes:
            if token_clean not in token_to_number_token_indexes:
                token_to_number_token_indexes[token_clean] = len(token_indexes)
            # Skip token with different indexes - it can be whitespace indexes
            elif token_to_number_token_indexes[token_clean] != len(token_indexes):
                continue

        predictions.append(PredictionBase(token=token_clean, score=prediction_token.total_score))

    return predictions


def convert_prediction_tokens_to_aggregation(
    tokenizer: PreTrainedTokenizer, all_prediction_tokens: List[List[PredictionToken]]
) -> List[PredictionForAggregation]:
    aggregations = []
    for prediction_tokens in tqdm(all_prediction_tokens, desc="Tokenizing predictions"):
        aggregations.append(
            PredictionForAggregation(
                type_score="probability", predictions=convert_prediction_tokens(tokenizer, prediction_tokens)
            )
        )
    return aggregations


def tokenize_clm_lm_gap_text(
    tokenizer: PreTrainedTokenizer, lm_gap_text: List[LMGapContextLine], buffer_length: int = 5
) -> List[LMGapPredictionLine]:
    prediction_text = []
    counter_tokens_length: CounterType = Counter()
    # Use BOS token if tokenized text is empty
    begging_text_for_empty = [tokenizer.bos_token_id]
    assert begging_text_for_empty != [None], "Cannot get BOS token for predicting empty text!"

    for lm_gap_context_line in tqdm(lm_gap_text, desc="Tokenize LM-GAP text", total=len(lm_gap_text)):
        line = lm_gap_context_line.left_context
        tokenized_line = tokenizer.tokenize(line)
        counter_tokens_length.update([len(tokenized_line)])

        prediction_text.append(
            LMGapPredictionLine(
                left_context_token_indexes=tokenizer.convert_tokens_to_ids(tokenized_line)
                if tokenized_line
                else begging_text_for_empty,
                right_context_token_indexes=[],
            )
        )

    max_token_length = tokenizer.max_len_single_sentence - buffer_length
    total_tokens = sum([v for k, v in counter_tokens_length.most_common() if k >= max_token_length])
    if total_tokens:
        LOGGER.warning(
            f"Found {total_tokens:,d} lines with {max_token_length}+ (with buffer: {buffer_length}) tokens,"
            f" where total line is: {len(lm_gap_text):,d}"
        )

    LOGGER.info(f"Tokenized {len(prediction_text):,d} lines")
    return prediction_text


def prepare_clm_data_loader(tokenized_text: List[LMGapPredictionLine], tokenizer: PreTrainedTokenizer) -> DataLoader:
    dataset_json = []
    for data in tqdm(tokenized_text, desc="Preprocessing text", total=len(tokenized_text)):
        dataset_json.append({"text": tokenizer.decode(data.left_context_token_indexes)})

    def tokenize_fn(examples: LazyBatch) -> BatchEncoding:
        return tokenizer(text=examples["text"])

    dataset = Dataset.from_list(dataset_json)
    del dataset_json
    dataset = dataset.map(
        tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False, desc="Preparing dataset"
    )

    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
        shuffle=False,
        drop_last=False,
    )


def prepare_clm_data_loader_for_next_prediction(
    tokenizer: PreTrainedTokenizer, input_with_predicted_tokens: InputWithPredictionTokens, batch_size: int
) -> DataLoader:
    dataset = Dataset.from_list(
        [
            {
                "input_ids": input_with_predicted_tokens.input_indexes + predicted_token.token_indexes,
                "index_predicted_tokens": i,
            }
            for i, predicted_token in enumerate(input_with_predicted_tokens.predicted_tokens)
        ]
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
        shuffle=False,
        drop_last=False,
    )
