import logging
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from datasets.arrow_dataset import Dataset
from datasets.formatting.formatting import LazyBatch
from torch.utils.data import DataLoader
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorWithPadding, default_data_collator

from phd_model_evaluations.train.data_collator.data_collator_with_labels_padding import DataCollatorWithLabelsPadding
from phd_model_evaluations.utils.collator_utils import get_collator_class_list_from_model_name, get_special_collator
from phd_model_evaluations.utils.dataset_utils import get_dataset_from_text
from phd_model_evaluations.utils.model_utils import get_sequence_length

LOGGER = logging.getLogger(__name__)


def get_sequence_length_and_padding_strategy(
    model_name: str, tokenizer: PreTrainedTokenizer, sequence_length: Optional[int], pad_to_max_length: bool
) -> Tuple[int, Union[str, bool], Optional[Callable[[List[Any]], Any]]]:
    sequence_length = get_sequence_length(sequence_length, tokenizer)
    padding_strategy: Union[str, bool]
    # Pad to the sequence length or to the maximum length in the batch
    if pad_to_max_length:
        padding_strategy, data_collator = "max_length", default_data_collator
    else:
        # It will use data collator base on tokenization
        padding_strategy, data_collator = False, None

    additional_collator_class_list = get_collator_class_list_from_model_name(model_name)
    if additional_collator_class_list is not None:
        assert data_collator is not None, "Cannot add collator to default collator!"
        data_collator = get_special_collator(data_collator, additional_collator_class_list)

    return sequence_length, padding_strategy, data_collator


def get_tokenized_dataset_text(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    sequence_length: int,
    padding_strategy: Union[str, bool] = False,
    truncation: bool = False,
) -> Dataset:
    def tokenize_fn(examples: LazyBatch) -> BatchEncoding:
        return tokenizer(
            text=examples["text"],
            padding=padding_strategy,
            max_length=sequence_length,
            truncation=truncation,
        )

    return dataset.map(
        tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False, desc="Tokenizing text"
    )


def prepare_data_loader_for_text(
    file_path: Path,
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    sequence_length: int,
    pad_to_max_length: bool = False,
    join_examples: bool = False,
) -> DataLoader:
    assert (
        sequence_length <= tokenizer.model_max_length
    ), f"Given length ({sequence_length}) cannot be greater than model length ({tokenizer.model_max_length})"
    data_collator: Union[DataCollatorWithLabelsPadding, Optional[Callable]]
    sequence_length, padding_strategy, data_collator = get_sequence_length_and_padding_strategy(
        model_name, tokenizer, sequence_length, pad_to_max_length
    )

    def group_joined_text_fn(examples: LazyBatch) -> Dict:
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples["input_ids"])
        if total_length >= sequence_length:
            total_length = (total_length // sequence_length) * sequence_length
        result = {
            k: [t[i : i + sequence_length] for i in range(0, total_length, sequence_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = deepcopy(result["input_ids"])
        return result

    def prepare_labels_fn(examples: LazyBatch) -> Dict:
        examples["labels"] = deepcopy(examples["input_ids"])
        return {k: v for k, v in examples.items()}

    dataset = get_dataset_from_text(file_path)

    LOGGER.info(f"Processing text data with sequence length: {sequence_length}")
    if join_examples:
        LOGGER.info("Joining text examples into one")
        data_collator = default_data_collator
        dataset = get_tokenized_dataset_text(dataset, tokenizer, sequence_length)
        dataset = dataset.map(group_joined_text_fn, batched=True, load_from_cache_file=False, desc="Grouping text")
    else:
        dataset = get_tokenized_dataset_text(
            dataset, tokenizer, sequence_length, padding_strategy=padding_strategy, truncation=True
        )
        dataset = dataset.map(prepare_labels_fn, batched=True, load_from_cache_file=False, desc="Preparing labels")
        # Enable padding base on sequence length in the batch
        if data_collator is None:
            data_collator = DataCollatorWithLabelsPadding(DataCollatorWithPadding(tokenizer))

    additional_collator_class_list = get_collator_class_list_from_model_name(model_name)
    if additional_collator_class_list is not None:
        assert data_collator is not None, "Cannot add collator to default collator!"
        data_collator = get_special_collator(data_collator, additional_collator_class_list)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        drop_last=False,
    )
