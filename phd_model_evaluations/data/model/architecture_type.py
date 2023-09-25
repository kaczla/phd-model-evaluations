from enum import Enum
from typing import Any, Dict


class ArchitectureType(str, Enum):
    encoder = "encoder"
    decoder = "decoder"
    encoder_decoder = "encoder_decoder"

    def __str__(self) -> str:
        return self.value


def get_architecture_type(config_dict: Dict[str, Any], model_name: str) -> ArchitectureType:
    if config_dict.get("architectures"):
        if any("ForMaskedLM" in i_config for i_config in config_dict["architectures"]):
            return ArchitectureType.encoder

        elif any("LMHeadModel" in i_config or "ForCausalLM" in i_config for i_config in config_dict["architectures"]):
            return ArchitectureType.decoder

        elif any("ForConditionalGeneration" in i_config for i_config in config_dict["architectures"]):
            return ArchitectureType.encoder_decoder

    if "model_type" in config_dict:
        model_type = config_dict["model_type"].lower()

        if "bert" in model_type or "longformer" in model_type:
            return ArchitectureType.encoder

        elif "gpt2" in model_type:
            return ArchitectureType.decoder

    raise RuntimeError(f"Cannot detect architecture type for: {model_name}")
