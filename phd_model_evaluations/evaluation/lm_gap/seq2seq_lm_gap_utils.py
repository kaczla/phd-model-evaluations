import logging
from collections import Counter
from typing import Counter as CounterType
from typing import Dict, List, Set, Tuple

from datasets import Dataset
from datasets.formatting.formatting import LazyBatch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, ByT5Tokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizer

from phd_model_evaluations.data.input_with_seq2seq_prediction_tokens import InputWithSeq2SeqPredictionTokens
from phd_model_evaluations.data.lm_gap.lm_gap_context_line import LMGapContextLine
from phd_model_evaluations.data.lm_gap.lm_gap_prediction_line import LMGapPredictionLine
from phd_model_evaluations.utils.type_utils import SEQ2SEQ_MODEL_TYPES

LOGGER = logging.getLogger(__name__)


def get_token_to_index(tokenizer: PreTrainedTokenizer) -> Dict[str, int]:
    data: Dict[str, int] = (
        tokenizer.special_tokens_encoder if isinstance(tokenizer, ByT5Tokenizer) else tokenizer.get_vocab()
    )
    return data


def get_seq2seq_predict_token_index(tokenizer: PreTrainedTokenizer) -> Tuple[str, int]:
    raw_vocab = get_token_to_index(tokenizer)
    additional_special_tokens = tokenizer.additional_special_tokens

    for predict_token in ["<extra_id_0>", "▁<extra_id_0>"]:
        if predict_token in additional_special_tokens or predict_token in raw_vocab:
            predict_token_index = tokenizer.convert_tokens_to_ids(predict_token)
            assert isinstance(
                predict_token_index, int
            ), f"Expected single index for prediction token, but got: {predict_token_index}"
            LOGGER.info(f"Using seq2seq prediction token: {predict_token} (index: {predict_token_index})")
            return predict_token, predict_token_index

    raise RuntimeError(
        f"Cannot get seq2seq prediction token for {tokenizer.__class__.__name__} tokenizer,"
        f" found special tokens: {additional_special_tokens}"
    )


def get_special_token_indexes(tokenizer: PreTrainedTokenizer) -> Set[int]:
    special_token_indexes = set(tokenizer.all_special_ids)

    raw_vocab = get_token_to_index(tokenizer)
    for i in range(100):
        for prefix_token, suffix_token in [("<extra_id_", ">"), ("▁<extra_id_", ">")]:
            token = f"{prefix_token}{i}{suffix_token}"
            if token in raw_vocab:
                special_token_indexes.add(raw_vocab[token])

    return special_token_indexes


def tokenize_seq2seq_lm_gap_text(
    tokenizer: PreTrainedTokenizer, lm_gap_text: List[LMGapContextLine], predict_token: str, buffer_length: int = 5
) -> List[LMGapPredictionLine]:
    prediction_text = []
    counter_tokens_length: CounterType = Counter()

    for lm_gap_context_line in tqdm(lm_gap_text, desc="Tokenize LM-GAP text", total=len(lm_gap_text)):
        line = lm_gap_context_line.left_context + " " + predict_token + lm_gap_context_line.right_context
        tokenized_line = tokenizer.tokenize(line)
        counter_tokens_length.update([len(tokenized_line)])
        index = tokenized_line.index(predict_token)
        assert index >= 0, "Cannot get seq2seq prediction token after tokenization"
        left_context = tokenized_line[:index]
        right_context = tokenized_line[index + 1 :]

        prediction_text.append(
            LMGapPredictionLine(
                left_context_token_indexes=tokenizer.convert_tokens_to_ids(left_context),
                right_context_token_indexes=tokenizer.convert_tokens_to_ids(right_context),
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


def prepare_seq2seq_data_loader(
    tokenized_text: List[LMGapPredictionLine],
    model: SEQ2SEQ_MODEL_TYPES,
    tokenizer: PreTrainedTokenizer,
    predict_token_index: int,
    add_padding_token: bool = True,
) -> DataLoader:
    dataset_json = []
    for data in tqdm(tokenized_text, desc="Preprocessing text", total=len(tokenized_text)):
        dataset_json.append(
            {
                "text": tokenizer.decode(
                    data.left_context_token_indexes + [predict_token_index] + data.right_context_token_indexes
                )
            }
        )

    LOGGER.info(f"Adding padding token to decoder input: {add_padding_token}")
    pad_token_id = tokenizer.pad_token_id
    decoder_input_ids = [[pad_token_id, predict_token_index]] if add_padding_token else [[predict_token_index]]
    decoder_attention_mask = [[1, 1]] if add_padding_token else [[1]]

    def tokenize_fn(examples: LazyBatch) -> BatchEncoding:
        batch = tokenizer(text=examples["text"])
        total_batch_examples = len(batch["input_ids"])
        batch["decoder_input_ids"] = decoder_input_ids * total_batch_examples
        batch["decoder_attention_mask"] = decoder_attention_mask * total_batch_examples
        return batch

    dataset = Dataset.from_list(dataset_json)
    del dataset_json
    dataset = dataset.map(
        tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False, desc="Preparing dataset"
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id, return_tensors="pt"
    )

    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
    )


def prepare_seq2seq_dataloader_for_next_prediction(
    model: SEQ2SEQ_MODEL_TYPES,
    tokenizer: PreTrainedTokenizer,
    input_with_seq2seq_predicted_tokens: InputWithSeq2SeqPredictionTokens,
    batch_size: int,
) -> DataLoader:
    dataset = Dataset.from_list(
        [
            {
                "input_ids": input_with_seq2seq_predicted_tokens.input_indexes,
                "decoder_input_ids": input_with_seq2seq_predicted_tokens.decoder_input_indexes
                + predicted_token.token_indexes,
                "index_predicted_tokens": i,
            }
            for i, predicted_token in enumerate(input_with_seq2seq_predicted_tokens.predicted_tokens)
        ]
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id, return_tensors="pt"
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
    )
