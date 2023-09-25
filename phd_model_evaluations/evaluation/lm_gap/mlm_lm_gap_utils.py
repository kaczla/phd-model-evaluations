import logging
from collections import Counter
from typing import Counter as CounterType
from typing import Dict, List, Tuple, Union

from datasets.arrow_dataset import Dataset
from datasets.formatting.formatting import LazyBatch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedTokenizer, PreTrainedTokenizerFast

from phd_model_evaluations.data.lm_gap.lm_gap_context_line import LMGapContextLine
from phd_model_evaluations.data.lm_gap.lm_gap_prediction_line import LMGapPredictionLine
from phd_model_evaluations.data.token_prediction_mlm import TokenPredictionMLM
from phd_model_evaluations.data.tokenization.sub_word_type_detector import SubWordTypeDetector
from phd_model_evaluations.data.tokenized_token import TokenizedToken
from phd_model_evaluations.utils.common_utils import get_value_from_directory_items
from phd_model_evaluations.utils.tokenization_utils import get_sub_word_type_detector

LOGGER = logging.getLogger(__name__)


def get_tokens_for_mlm_lm_gap(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], max_token_size: int, smart_tokens: bool = False
) -> List[TokenPredictionMLM]:
    LOGGER.info(f"Getting tokens for max size: {max_token_size} ...")
    if 0 < max_token_size > 1:
        raise NotImplementedError("Token size greater than 1 is not implemented!")

    context_left_token_indexes = tokenizer.encode(" This a simple", add_special_tokens=False)
    context_right_token_indexes = tokenizer.encode(" dummy sentence!", add_special_tokens=False)

    invalid_token_indexes = []
    special_token_indexes = set(tokenizer.all_special_ids)
    sub_word_type_detector = get_sub_word_type_detector(tokenizer) if smart_tokens else SubWordTypeDetector()

    tokens: List[TokenPredictionMLM] = [
        TokenPredictionMLM(
            token_size=i + 1,
            prefix_indexes=[],
            prefix_tokens=[],
            token_indexes=[],
            token_str_list=[],
        )
        for i in range(max_token_size)
    ]
    for token, token_index in sorted(tokenizer.get_vocab().items(), key=get_value_from_directory_items):
        # Skip special tokens
        if token_index in special_token_indexes:
            continue

        # Skip inner tokens
        if smart_tokens and sub_word_type_detector.is_inner_word(token):
            continue

        # Skip whitespace tokens
        if not tokenizer.decode([token_index]).strip():
            continue

        # Check tokenization decoding and encoding generate the same sentence
        input_token_indexes = context_left_token_indexes + [token_index] + context_right_token_indexes
        input_decoded = tokenizer.decode(input_token_indexes)
        input_encoded = tokenizer.encode(input_decoded, add_special_tokens=False)
        # Skip token if generate different tokens after decoding
        if input_token_indexes != input_encoded:
            invalid_token_indexes.append(token_index)
            continue

        tokens[0].token_indexes.append(token_index)
        tokens[0].token_str_list.append(token)

    LOGGER.debug(f"Skipped {len(invalid_token_indexes)} tokens")
    LOGGER.info(f"Got {len(tokens[0].token_str_list)} tokens for MLM prediction with length equal to 1")
    assert sum([len(token.token_str_list) for token in tokens]) > 0, "Missing tokens for MLM prediction"
    return tokens


def tokenize_mlm_lm_gap_text(
    tokenizer: PreTrainedTokenizer, lm_gap_text: List[LMGapContextLine], buffer_length: int = 5
) -> List[LMGapPredictionLine]:
    mask_token = tokenizer.mask_token
    prediction_text = []
    counter_tokens_length: CounterType = Counter()

    for i, lm_gap_context_line in tqdm(enumerate(lm_gap_text), desc="Tokenize LM-GAP text", total=len(lm_gap_text)):
        line = lm_gap_context_line.left_context + " " + mask_token + " " + lm_gap_context_line.right_context
        tokenized_line = tokenizer.tokenize(line)
        counter_tokens_length.update([len(tokenized_line)])

        mask_token_indexes_in_text = [j for j, token_str in enumerate(tokenized_line) if mask_token in token_str]
        assert (
            len(mask_token_indexes_in_text) == 1
        ), f"Using more than 1 mask token is not supported! Found {len(mask_token_indexes_in_text)} in line: {i + 1}"
        mask_index_text = mask_token_indexes_in_text[0]

        prediction_text.append(
            LMGapPredictionLine(
                left_context_token_indexes=tokenizer.convert_tokens_to_ids(tokenized_line[:mask_index_text]),
                right_context_token_indexes=tokenizer.convert_tokens_to_ids(tokenized_line[mask_index_text + 1 :]),
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


def get_tokens_prediction_mlm(
    tokenized_tokens: List[TokenizedToken],
) -> List[TokenPredictionMLM]:
    """
    Get list of MLM tokens to predict grouped by "token size" and "prefix tokens".

    "Token size" is the number of tokens after tokenization, e.g. token "1923" tokenized
    to 2 tokens: "19" "23" - token size is equal to 2.

    "Prefix tokens" is the list of tokens after tokenization without last token, e.g. token "1835" tokenized
    to 1 token "1835" doesn't have prefix (will be empty list), token "1923" tokenized to 2 tokens: "19" "23"
    have 1 token prefix: "19", and so on.

    Grouping by "token size" and "prefix tokens" allows to group the tokens with the same number of tokens
    and prefix tokens in the same group, e.g. tokens "1923" (tokenized to "19" "23") and "1955"
    (tokenized to "19" "55") will be in the same group (token size is equal to 2 and prefix tokens
    is equal to "19").

    Parameters
        tokenized_tokens: list of tokenized tokens to group

    Returns:
        list of tokenized tokens grouped by "token size" and "prefix tokens"

    """
    # Create dictionary with the prefixes list (as tuple - which can be indexed in dictionary)
    prefix_tokens_to_token_prediction_mlm: Dict[Tuple[str, ...], TokenPredictionMLM] = {}
    for tokenized_token in tokenized_tokens:
        token_index = tokenized_token.indexes[-1]
        token_str = tokenized_token.tokens[-1]
        prefix_indexes = tokenized_token.indexes[:-1]
        prefix_tokens = tokenized_token.tokens[:-1]

        # Skip prefix for single token - empty tuple
        key_index: Tuple[str, ...]
        if len(tokenized_token.tokens) == 1:
            key_index = ()
        # Get first token only (ignore second/last token)
        elif len(tokenized_token.tokens) == 2:
            key_index = (tokenized_token.tokens[0],)
        # Ignore last token
        else:
            key_index = tuple(tokenized_token.tokens[:-1])

        # Add token with the same prefix
        if key_index in prefix_tokens_to_token_prediction_mlm:
            prefix_tokens_to_token_prediction_mlm[key_index].token_indexes.append(token_index)
            prefix_tokens_to_token_prediction_mlm[key_index].token_str_list.append(token_str)
        # Add new group
        else:
            prefix_tokens_to_token_prediction_mlm[key_index] = TokenPredictionMLM(
                token_size=len(prefix_indexes) + 1,
                prefix_indexes=prefix_indexes,
                prefix_tokens=prefix_tokens,
                token_indexes=[token_index],
                token_str_list=[token_str],
            )

    return list(prefix_tokens_to_token_prediction_mlm.values())


def prepare_mlm_data_loader(
    tokenized_text: List[LMGapPredictionLine],
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
) -> DataLoader:
    dataset_json = []
    for data in tqdm(tokenized_text, desc="Preprocessing text", total=len(tokenized_text)):
        text_line = tokenizer.decode(
            data.left_context_token_indexes + [tokenizer.mask_token_id] + data.right_context_token_indexes
        )
        dataset_json.append({"text": text_line})

    def tokenize_fn(examples: LazyBatch) -> BatchEncoding:
        return tokenizer(text=examples["text"])

    dataset = Dataset.from_list(dataset_json)
    del dataset_json
    dataset = dataset.map(
        tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False, desc="Preparing dataset"
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
        shuffle=False,
        drop_last=False,
    )
