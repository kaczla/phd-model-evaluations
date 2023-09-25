from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_name: str = field(metadata={"help": "Name of model from huggingface.co/models or model path."})
    model_human_name: Optional[str] = field(
        default=None,
        metadata={"help": "Human model name, readable for the human used in file names, aggregations and statistics."},
    )
    model_low_cpu_mem_usage: bool = field(
        default=False,
        metadata={"help": "Use the lowest memory as is possible."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Use the fast tokenizer (backed by the tokenizers library) or not."},
    )
    add_prefix_space: bool = field(
        default=False,
        metadata={
            "help": "Add space at the beginning of the tokenizer text, if not enabled will be used"
            " default option for tokenizer."
        },
    )
    batch_size: int = field(
        default=5,
        metadata={
            "help": "Batch size - number lines in each batch. Will be ignored if is other argument"
            " for training/evaluation batch size exists."
        },
    )
    max_batch_size_with_gradient_accumulation: Optional[int] = field(
        default=None,
        metadata={
            "help": "Final batch size with gradient accumulation, if passed `gradient_accumulation_steps` argument"
            " will be adjusted with `per_device_train_batch_size` argument to reach given value,"
            " e.g. if passed value is 30 and batch size is 4 then gradient accumulation will be set to 5."
        },
    )
    sequence_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum token length for input text after tokenization."
                " Sequences longer than this will be truncated, sequences shorter will be padded."
                " If missing it will be used sequence length from the model."
            )
        },
    )
    target_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum token length for target text after tokenization (only seq2seq models)."
                " Sequences longer than this will be truncated, sequences shorter will be padded."
                " If missing it will be used sequence length from the model."
            )
        },
    )
    dynamic_generation_length: bool = field(
        default=True,
        metadata={
            "help": "Use generation length base on maximum length from target/label data in used dataset"
            " (only seq2seq models). If enable, it will update value in `generation_max_length` argument."
            " Otherwise will use value passed in `generation_max_length` argument."
        },
    )
    device: int = field(
        default=0,
        metadata={"help": "Number of CUDA device, -1 is CPU."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Pad to sequence length or to the maximum length in the batch - it will choose the smaller one."
        },
    )
    early_stopping_patience: int = field(
        default=0,
        metadata={
            "help": "How many times evaluation metric can be worst than the best metric before stopping training,"
            " 0 means disabled."
        },
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "How much the specified metric must improve to satisfy early stopping conditions."},
    )
    use_custom_model: bool = field(
        default=False,
        metadata={
            "help": "Use custom model for classification or regression (only for encoder or decoder models)."
            " It will use custom implementation instead of implementation from `transformers` library,"
            " it also allows train model which are not supported in `transformers` library."
        },
    )

    def get_model_name(self) -> str:
        if self.model_human_name is not None:
            return self.model_human_name

        return self.model_name
