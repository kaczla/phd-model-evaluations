import logging
from typing import Any, Dict, List, Tuple

from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizer

from phd_model_evaluations.data.model.architecture_type import ArchitectureType, get_architecture_type
from phd_model_evaluations.data.model.model_parameters import ModelParameters
from phd_model_evaluations.data.results.table_data import TableData
from phd_model_evaluations.utils.model_names import MAP_MODEL_NAME_TO_HUMAN_NAME
from phd_model_evaluations.utils.model_utils import (
    get_sequence_length_from_config_or_predefined_list,
    is_unlimited_sequence_length,
)

LOGGER = logging.getLogger(__name__)


def check_if_is_empty(value_first: int, value_second: int, name: str) -> None:
    if value_first == 0 and value_second == 0:
        raise RuntimeError(f"{name} should not be equal to 0!")


def get_value_from_config_dict(config_dict: Dict[str, Any], key_names: List[str], default_value: int) -> int:
    value = default_value

    for key in key_names:
        if key in config_dict:
            value_raw = config_dict[key]
            if value_raw is None:
                continue

            value = int(value_raw)
            break

    return value


def get_number_layers(config_dict: Dict[str, Any], architecture_type: ArchitectureType) -> Tuple[int, int]:
    encoder_layers = 0
    decoder_layers = get_value_from_config_dict(config_dict, ["num_decoder_layers"], 0)
    layers = get_value_from_config_dict(config_dict, ["num_hidden_layers", "num_layers", "n_layer", "n_layers"], 0)

    if architecture_type == ArchitectureType.decoder:
        decoder_layers = layers
    else:
        encoder_layers = layers

    check_if_is_empty(encoder_layers, decoder_layers, "Number of layers")
    return encoder_layers, decoder_layers


def get_hidden_size(config_dict: Dict[str, Any], architecture_type: ArchitectureType) -> Tuple[int, int]:
    encoder_hidden_size = 0
    decoder_hidden_size = 0
    hidden_size = get_value_from_config_dict(config_dict, ["hidden_size", "d_model", "n_embd", "dim"], 0)

    if architecture_type == ArchitectureType.decoder:
        decoder_hidden_size = hidden_size
    elif architecture_type == ArchitectureType.encoder:
        encoder_hidden_size = hidden_size
    else:
        encoder_hidden_size = hidden_size
        decoder_hidden_size = hidden_size

    check_if_is_empty(encoder_hidden_size, decoder_hidden_size, "Hidden size of model")
    return encoder_hidden_size, decoder_hidden_size


def get_feedforward_hidden_size(config_dict: Dict[str, Any], architecture_type: ArchitectureType) -> Tuple[int, int]:
    encoder_feedforward_hidden_size = 0
    decoder_feedforward_hidden_size = 0
    feedforward_hidden_size = get_value_from_config_dict(
        config_dict, ["intermediate_size", "d_ff", "n_ctx", "ffn_dim", "hidden_dim", "n_inner"], 0
    )

    if architecture_type == ArchitectureType.decoder:
        decoder_feedforward_hidden_size = feedforward_hidden_size
    elif architecture_type == ArchitectureType.encoder:
        encoder_feedforward_hidden_size = feedforward_hidden_size
    else:
        encoder_feedforward_hidden_size = feedforward_hidden_size
        decoder_feedforward_hidden_size = feedforward_hidden_size

    if (
        encoder_feedforward_hidden_size == 0
        and decoder_feedforward_hidden_size == 0
        and "GPTNeoForCausalLM" in config_dict.get("architectures", {})
    ):
        _, decoder_hidden_size = get_hidden_size(config_dict, architecture_type)
        decoder_feedforward_hidden_size = 4 * decoder_hidden_size

    check_if_is_empty(encoder_feedforward_hidden_size, decoder_feedforward_hidden_size, "Hidden size of FFN")
    return encoder_feedforward_hidden_size, decoder_feedforward_hidden_size


def get_number_attention_heads(config_dict: Dict[str, Any], architecture_type: ArchitectureType) -> Tuple[int, int]:
    encoder_number_attention_heads = 0
    decoder_number_attention_heads = 0
    number_attention_heads = get_value_from_config_dict(
        config_dict, ["num_attention_heads", "num_heads", "n_head", "n_heads"], 0
    )

    if architecture_type == ArchitectureType.decoder:
        decoder_number_attention_heads = number_attention_heads
    elif architecture_type == ArchitectureType.encoder:
        encoder_number_attention_heads = number_attention_heads
    else:
        encoder_number_attention_heads = number_attention_heads
        decoder_number_attention_heads = number_attention_heads

    check_if_is_empty(encoder_number_attention_heads, decoder_number_attention_heads, "Number of attention heads")
    return encoder_number_attention_heads, decoder_number_attention_heads


def get_sequence_length(model_name: str, config: PretrainedConfig, tokenizer: PreTrainedTokenizer) -> int:
    sequence_length: int = tokenizer.model_max_length

    if is_unlimited_sequence_length(tokenizer):
        sequence_length_from_config = get_sequence_length_from_config_or_predefined_list(model_name, config)

        if sequence_length_from_config is None:
            LOGGER.info(f"Not found sequence length for: {model_name}")
            return 0

        return sequence_length_from_config

    return sequence_length


def get_model_parameters(model_name: str) -> ModelParameters:
    LOGGER.info(f"Generating model parameters for: {model_name}")
    human_name = MAP_MODEL_NAME_TO_HUMAN_NAME.get(model_name, "")

    config: PretrainedConfig = AutoConfig.from_pretrained(model_name)
    config_dict = config.to_dict()
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

    architecture_type = get_architecture_type(config_dict, model_name)
    encoder_layers, decoder_layers = get_number_layers(config_dict, architecture_type)
    encoder_hidden_size, decoder_hidden_size = get_hidden_size(config_dict, architecture_type)
    encoder_feedforward_hidden_size, decoder_feedforward_hidden_size = get_feedforward_hidden_size(
        config_dict, architecture_type
    )
    encoder_number_attention_heads, decoder_number_attention_heads = get_number_attention_heads(
        config_dict, architecture_type
    )
    sequence_length = get_sequence_length(model_name, config, tokenizer)

    return ModelParameters(
        name=model_name,
        human_name=human_name,
        architecture_type=architecture_type,
        number_parameters=0,
        encoder_number_layers=encoder_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_feedforward_hidden_size=encoder_feedforward_hidden_size,
        encoder_number_attention_heads=encoder_number_attention_heads,
        encoder_attention_size=0
        if encoder_number_attention_heads == 0
        else encoder_hidden_size // encoder_number_attention_heads,
        decoder_number_layers=decoder_layers,
        decoder_hidden_size=decoder_hidden_size,
        decoder_feedforward_hidden_size=decoder_feedforward_hidden_size,
        decoder_number_attention_heads=decoder_number_attention_heads,
        decoder_attention_size=0
        if decoder_number_attention_heads == 0
        else decoder_hidden_size // decoder_number_attention_heads,
        sequence_length=sequence_length,
        vocab_size=tokenizer.vocab_size,
    )


def generate_model_parameters(model_names: List[str]) -> List[ModelParameters]:
    return [get_model_parameters(model_name) for model_name in model_names]


def convert_model_parameters_to_table_data(model_parameters_list: List[ModelParameters]) -> TableData:
    model_names = []
    one_line_row_names = []
    row_data = []
    for model_parameters in model_parameters_list:
        model_name = model_parameters.human_name
        model_names.append(model_name)
        one_line_row_names.append(model_name)
        row_data.append(model_parameters.get_table_row_with_model_name())
        model_names.append(f"{model_name} parameters")
        row_data.append(model_parameters.get_table_row())

    return TableData(
        column_names=ModelParameters.get_table_header(),
        row_names=model_names,
        row_data=row_data,
        one_line_row_names=one_line_row_names,
        skip_row_name=True,
    )
