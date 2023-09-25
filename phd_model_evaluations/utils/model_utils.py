import json
import logging
from pathlib import Path
from sys import maxsize
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LongformerConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.trainer_utils import get_last_checkpoint

from phd_model_evaluations.data.arguments.model_arguments import ModelArguments
from phd_model_evaluations.data.model.architecture_type import ArchitectureType, get_architecture_type
from phd_model_evaluations.utils.type_utils import (
    CLM_MODEL_TYPES,
    MLM_MODEL_TYPES,
    MODEL_TYPES,
    MODEL_WITHOUT_LM_HEAD_TYPES,
    SEQ2SEQ_MODEL_TYPES,
)

LOGGER = logging.getLogger(__name__)

MODEL_NAME_TO_SEQUENCE_LENGTH: Dict[str, int] = {
    "google/mt5-small": 1024,
    "google/mt5-base": 1024,
    "google/mt5-large": 1024,
    "google/mt5-xl": 1024,
    "google/mt5-xxl": 1024,
    "google/byt5-small": 1024,
    "google/byt5-base": 1024,
    "google/byt5-large": 1024,
    "google/byt5-xl": 1024,
    "google/byt5-xxl": 1024,
}

MODEL_NAME_TO_ALLOWED_MISSING_WEIGHTS: Dict[str, List[str]] = {
    "openai-gpt": ["position_ids"],
}


def get_sequence_length_from_config_or_predefined_list(model_name: str, config: PretrainedConfig) -> Optional[int]:
    sequence_length = get_sequence_length_from_config(config)

    if sequence_length is None:
        sequence_length = get_sequence_length_from_predefined_list(model_name)

    return sequence_length


def get_sequence_length_from_config(config: PretrainedConfig) -> Optional[int]:
    sequence_length = None

    config_dict = config.to_dict()
    for key_to_check in ["max_position_embeddings", "n_positions"]:
        if key_to_check in config_dict:
            return int(config_dict[key_to_check])

    return sequence_length


def get_sequence_length_from_predefined_list(model_name: str) -> Optional[int]:
    if model_name in MODEL_NAME_TO_SEQUENCE_LENGTH:
        LOGGER.debug(f"Use sequence length from predefined list for: {model_name}")
        return MODEL_NAME_TO_SEQUENCE_LENGTH[model_name]

    return None


def is_unlimited_sequence_length(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> bool:
    is_unlimited: bool = tokenizer.model_max_length > maxsize
    return is_unlimited


def get_config(model_arguments: ModelArguments) -> PretrainedConfig:
    config: PretrainedConfig = AutoConfig.from_pretrained(model_arguments.model_name)
    return config


def get_tokenizer(model_arguments: ModelArguments) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """
    Load tokenizer.

    Args:
        model_arguments: model arguments used to initialize tokenizer

    Returns:
        loaded tokenizer

    """
    LOGGER.info(f"Loading tokenizer: {model_arguments.model_name}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_arguments.model_name,
        use_fast=model_arguments.use_fast_tokenizer,
        add_prefix_space=model_arguments.add_prefix_space,
    )
    LOGGER.info("Tokenizer loaded")

    if is_unlimited_sequence_length(tokenizer):
        config = get_config(model_arguments)
        model_max_length = get_sequence_length_from_config_or_predefined_list(model_arguments.model_name, config)

        if model_max_length is None:
            raise RuntimeError(f"Cannot detect sequence length of tokenizer: {tokenizer.__class__.__name__}")

        tokenizer.model_max_length = model_max_length
        LOGGER.info(f"Loaded max sequence length from config in tokenizer, sequence length: {model_max_length}")

    return tokenizer


def get_model_architecture(model_arguments: ModelArguments) -> ArchitectureType:
    """
    Get type of architecture (MLM, CLM or Seq2Seq).

    Args:
        model_arguments: model arguments used to initialize model

    Returns:
        type of architecture (MLM, CLM or Seq2Seq)

    """

    model_name = model_arguments.get_model_name()
    config = get_config(model_arguments)
    return get_architecture_type(config.to_dict(), model_name)


def get_model(model_arguments: ModelArguments) -> MODEL_TYPES:
    """
    Load any Transformer model.

    Args:
        model_arguments: model arguments used to initialize model

    Returns:
        loaded model

    """
    architecture_type = get_model_architecture(model_arguments)

    if architecture_type == ArchitectureType.encoder:
        return get_mlm_model(model_arguments)

    elif architecture_type == ArchitectureType.decoder:
        return get_clm_model(model_arguments)

    return get_seq2seq_model(model_arguments)


def get_mlm_model(model_arguments: ModelArguments) -> MLM_MODEL_TYPES:
    """
    Load MLM (encoder) Transformer model.

    Args:
        model_arguments: model arguments used to initialize model

    Returns:
        loaded model

    """
    LOGGER.info(f"Loading MLM model: {model_arguments.model_name}")
    model, loading_info = AutoModelForMaskedLM.from_pretrained(
        model_arguments.model_name,
        low_cpu_mem_usage=model_arguments.model_low_cpu_mem_usage,
        output_loading_info=True,
    )
    check_missing_keys_in_loading_info(model_arguments.model_name, loading_info)
    torch_device = get_torch_device(model_arguments.device)
    model.to(device=torch_device)
    LOGGER.info(f"Model loaded from class: {model.__class__.__name__}")
    return model


def get_clm_model(model_arguments: ModelArguments) -> CLM_MODEL_TYPES:
    """
    Load CLM (decoder) Transformer model.

    Args:
        model_arguments: model arguments used to initialize model

    Returns:
        loaded model

    """
    LOGGER.info(f"Loading CLM model: {model_arguments.model_name}")
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_arguments.model_name,
        low_cpu_mem_usage=model_arguments.model_low_cpu_mem_usage,
        output_loading_info=True,
    )
    check_missing_keys_in_loading_info(model_arguments.model_name, loading_info)
    torch_device = get_torch_device(model_arguments.device)
    model.to(device=torch_device)
    LOGGER.info(f"Model loaded from class: {model.__class__.__name__}")
    return model


def get_seq2seq_model(model_arguments: ModelArguments) -> SEQ2SEQ_MODEL_TYPES:
    """
    Load seq2seq (encoder-decoder) Transformer model.

    Args:
        model_arguments: model arguments used to initialize model

    Returns:
        loaded model

    """
    LOGGER.info(f"Loading seq2seq model: {model_arguments.model_name}")
    model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(
        model_arguments.model_name,
        low_cpu_mem_usage=model_arguments.model_low_cpu_mem_usage,
        output_loading_info=True,
    )
    check_missing_keys_in_loading_info(model_arguments.model_name, loading_info)
    torch_device = get_torch_device(model_arguments.device)
    model.to(device=torch_device)
    LOGGER.info(f"Model loaded from class: {model.__class__.__name__}")
    return model


def get_auto_model(model_arguments: ModelArguments) -> MODEL_WITHOUT_LM_HEAD_TYPES:
    """
    Load Transformer model without LM-Head.

    Args:
        model_arguments: model arguments used to initialize model

    Returns:
        loaded model

    """
    LOGGER.info(f"Loading auto model: {model_arguments.model_name}")
    model, loading_info = AutoModel.from_pretrained(
        model_arguments.model_name,
        low_cpu_mem_usage=model_arguments.model_low_cpu_mem_usage,
        output_loading_info=True,
    )
    check_missing_keys_in_loading_info(model_arguments.model_name, loading_info)
    torch_device = get_torch_device(model_arguments.device)
    model.to(device=torch_device)
    LOGGER.info(f"Model loaded from class: {model.__class__.__name__}")
    return model


def get_torch_device(device_id: int) -> torch.device:
    torch_device = torch.device("cpu") if device_id < 0 else torch.device(device_id)
    LOGGER.info(f"Using device: {torch_device}")
    return torch_device


def get_sequence_length(sequence_length: Optional[int], tokenizer: PreTrainedTokenizer) -> int:
    if sequence_length is None:
        LOGGER.info(f"Using sequence length from tokenizer: {tokenizer.model_max_length}")
        return int(tokenizer.model_max_length)

    if sequence_length > tokenizer.model_max_length:
        new_sequence_length = min(sequence_length, int(tokenizer.model_max_length))
        LOGGER.warning(
            f"The sequence_length passed ({sequence_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length})."
        )
        LOGGER.info(f"Using sequence length limited by tokenizer: {new_sequence_length}.")
        return new_sequence_length

    LOGGER.info(f"Using sequence length: {sequence_length}")
    return sequence_length


def get_last_checkpoint_directory(save_path: Union[str, Path]) -> Optional[str]:
    save_path = Path(save_path) if isinstance(save_path, str) else save_path
    if not save_path.exists():
        return None

    checkpoint_directory: Optional[str] = get_last_checkpoint(str(save_path))
    if checkpoint_directory is None:
        return None

    LOGGER.info(f"Using last checkpoint from: {checkpoint_directory}")
    return checkpoint_directory


def get_best_checkpoint_directory(save_path: Union[str, Path]) -> Optional[str]:
    save_path = Path(save_path) if isinstance(save_path, str) else save_path
    if not save_path.exists():
        LOGGER.warning(f"Empty checkpoints, cannot find best checkpoint in: {save_path}")
        return None
    # Use trainer_state.json from current directory or from last checkpoint directory
    trainer_state_path = save_path / "trainer_state.json"
    if not trainer_state_path.exists():
        checkpoint_directory: Optional[str] = get_last_checkpoint(str(save_path))
        if checkpoint_directory is None:
            LOGGER.warning(f"Last checkpoint doesn't exist, cannot find best checkpoint in: {save_path}")
            return None

        trainer_state_path = save_path / Path(checkpoint_directory).name / "trainer_state.json"

    LOGGER.debug(f"Loading best metric from: {trainer_state_path}")
    trainer_state = json.loads(trainer_state_path.read_text())
    original_best_model_path = Path(trainer_state["best_model_checkpoint"])

    best_model_path = save_path / original_best_model_path.name
    LOGGER.info(f"Found best model in: {best_model_path}")
    return str(best_model_path)


def check_missing_keys_in_loading_info(model_name: str, loading_info: Dict) -> None:
    missing_keys = loading_info.get("missing_keys", [])
    if missing_keys:
        if missing_keys == MODEL_NAME_TO_ALLOWED_MISSING_WEIGHTS.get(model_name, []):
            LOGGER.debug(f"Loaded model with missing weights for: {missing_keys}")
            return

        raise RuntimeError(f"Cannot load load model: {model_name} - missing weights for: {missing_keys}")

    mismatched_keys = loading_info.get("mismatched_keys", [])
    if mismatched_keys:
        raise RuntimeError(f"Cannot load load model: {model_name} - missmatch weights for: {mismatched_keys}")


def check_padding_token(tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> None:
    if tokenizer.pad_token_id is None:
        LOGGER.info(f"Set PAD token as EOS token: {tokenizer.eos_token}")
        tokenizer._pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id


def update_config_model(model_arguments: ModelArguments, config: PretrainedConfig) -> PretrainedConfig:
    if isinstance(config, LongformerConfig) and model_arguments.sequence_length != config.attention_window[0]:
        old_attention_window = config.attention_window
        LOGGER.info(f"Changing attention window to: {model_arguments.sequence_length}")
        config.attention_window = [model_arguments.sequence_length for _ in range(len(config.attention_window))]
        LOGGER.info(f"Attention window changed from: {old_attention_window} to {config.attention_window}")

    return config


def get_best_model_path(model_path: str) -> str:
    best_model_path = get_best_checkpoint_directory(model_path)

    if best_model_path is None:
        raise RuntimeError(f"Cannot find best model in: {model_path} - missing trainer_state.json file")

    return best_model_path
