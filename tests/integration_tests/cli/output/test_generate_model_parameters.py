from pathlib import Path

from phd_model_evaluations.data.model.architecture_type import ArchitectureType
from phd_model_evaluations.data.model.model_parameters import ModelParameters
from phd_model_evaluations.output.generate_model_parameters import get_model_parameters


def test_get_model_parameters_roberta_base(path_roberta_base_model: Path) -> None:
    model_parameters = get_model_parameters(str(path_roberta_base_model))
    expected_model_parameters = ModelParameters(
        name=str(path_roberta_base_model),
        human_name="",
        architecture_type=ArchitectureType.encoder,
        number_parameters=0,
        encoder_number_layers=12,
        encoder_hidden_size=768,
        encoder_feedforward_hidden_size=3072,
        encoder_number_attention_heads=12,
        encoder_attention_size=64,
        decoder_number_layers=0,
        decoder_hidden_size=0,
        decoder_feedforward_hidden_size=0,
        decoder_number_attention_heads=0,
        decoder_attention_size=0,
        sequence_length=512,
        vocab_size=50265,
    )
    assert model_parameters == expected_model_parameters, "Invalid RoBERTa model parameters"


def test_get_model_parameters_gpt2_base(path_gpt2_base_model: Path) -> None:
    model_parameters = get_model_parameters(str(path_gpt2_base_model))
    expected_model_parameters = ModelParameters(
        name=str(path_gpt2_base_model),
        human_name="",
        architecture_type=ArchitectureType.decoder,
        number_parameters=0,
        encoder_number_layers=0,
        encoder_hidden_size=0,
        encoder_feedforward_hidden_size=0,
        encoder_number_attention_heads=0,
        encoder_attention_size=0,
        decoder_number_layers=12,
        decoder_hidden_size=768,
        decoder_feedforward_hidden_size=1024,
        decoder_number_attention_heads=12,
        decoder_attention_size=64,
        sequence_length=1024,
        vocab_size=50257,
    )
    assert model_parameters == expected_model_parameters, "Invalid GPT-2 model parameters"


def test_get_model_parameters_t5_small(path_t5_small_model: Path) -> None:
    model_parameters = get_model_parameters(str(path_t5_small_model))
    expected_model_parameters = ModelParameters(
        name=str(path_t5_small_model),
        human_name="",
        architecture_type=ArchitectureType.encoder_decoder,
        number_parameters=0,
        encoder_number_layers=6,
        encoder_hidden_size=512,
        encoder_feedforward_hidden_size=2048,
        encoder_number_attention_heads=8,
        encoder_attention_size=64,
        decoder_number_layers=6,
        decoder_hidden_size=512,
        decoder_feedforward_hidden_size=2048,
        decoder_number_attention_heads=8,
        decoder_attention_size=64,
        sequence_length=512,
        vocab_size=32100,
    )
    assert model_parameters == expected_model_parameters, "Invalid T5-small model parameters"


def test_get_model_parameters_t5_base(path_t5_base_model: Path) -> None:
    model_parameters = get_model_parameters(str(path_t5_base_model))
    expected_model_parameters = ModelParameters(
        name=str(path_t5_base_model),
        human_name="",
        architecture_type=ArchitectureType.encoder_decoder,
        number_parameters=0,
        encoder_number_layers=12,
        encoder_hidden_size=768,
        encoder_feedforward_hidden_size=3072,
        encoder_number_attention_heads=12,
        encoder_attention_size=64,
        decoder_number_layers=12,
        decoder_hidden_size=768,
        decoder_feedforward_hidden_size=3072,
        decoder_number_attention_heads=12,
        decoder_attention_size=64,
        sequence_length=512,
        vocab_size=32100,
    )
    assert model_parameters == expected_model_parameters, "Invalid T5-base model parameters"
