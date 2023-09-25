from typing import Dict, List

from pydantic import BaseModel

from phd_model_evaluations.data.model.architecture_type import ArchitectureType
from phd_model_evaluations.data.model.model_parameter_label import ModelParameterLabel


class ModelParameters(BaseModel):
    name: str
    human_name: str
    architecture_type: ArchitectureType
    number_parameters: int
    encoder_number_layers: int
    encoder_hidden_size: int
    encoder_feedforward_hidden_size: int
    encoder_number_attention_heads: int
    encoder_attention_size: int
    decoder_number_layers: int
    decoder_hidden_size: int
    decoder_feedforward_hidden_size: int
    decoder_number_attention_heads: int
    decoder_attention_size: int
    sequence_length: int
    vocab_size: int

    def get_table_row(self) -> Dict[str, str]:
        label_type = str(ModelParameterLabel.architecture_type)
        label_parameters = str(ModelParameterLabel.number_parameters)
        label_vocab = str(ModelParameterLabel.vocab_size)
        label_sequence = str(ModelParameterLabel.sequence_length)
        label_layers = str(ModelParameterLabel.number_layers)
        label_hidden = str(ModelParameterLabel.hidden_size)
        label_ffn = str(ModelParameterLabel.feedforward_hidden_size)
        label_att = str(ModelParameterLabel.number_attention_heads)
        label_att_dim = str(ModelParameterLabel.attention_size)

        data = {
            label_type: self.get_archetype_type_string(self.architecture_type),
            label_parameters: "-",
            label_vocab: self.get_human_size(self.vocab_size),
            label_sequence: self.get_human_size(self.sequence_length) if self.sequence_length > 0 else r"$\infty$",
        }

        if self.architecture_type == ArchitectureType.encoder:
            data[label_layers] = f"{self.encoder_number_layers}"
            data[label_hidden] = f"{self.encoder_hidden_size}"
            data[label_ffn] = f"{self.encoder_feedforward_hidden_size}"
            data[label_att] = f"{self.encoder_number_attention_heads}"
            data[label_att_dim] = f"{self.encoder_attention_size}"

        elif self.architecture_type == ArchitectureType.decoder:
            data[label_layers] = f"{self.decoder_number_layers}"
            data[label_hidden] = f"{self.decoder_hidden_size}"
            data[label_ffn] = f"{self.decoder_feedforward_hidden_size}"
            data[label_att] = f"{self.decoder_number_attention_heads}"
            data[label_att_dim] = f"{self.decoder_attention_size}"

        else:
            data[label_layers] = f"{self.encoder_number_layers}/{self.decoder_number_layers}"
            data[label_hidden] = f"{self.encoder_hidden_size}/{self.decoder_hidden_size}"
            data[label_ffn] = f"{self.encoder_feedforward_hidden_size}/{self.decoder_feedforward_hidden_size}"
            data[label_att] = f"{self.encoder_number_attention_heads}/{self.decoder_number_attention_heads}"
            data[label_att_dim] = f"{self.encoder_attention_size}/{self.decoder_attention_size}"

        return data

    def get_table_row_with_model_name(self) -> Dict[str, str]:
        return {str(ModelParameterLabel.architecture_type): self.human_name}

    @staticmethod
    def get_archetype_type_string(architecture_type: ArchitectureType) -> str:
        if architecture_type == ArchitectureType.encoder:
            # K = Koder
            return "K"

        elif architecture_type == ArchitectureType.decoder:
            # D = Dekoder
            return "D"

        # K-D = Koder-Dekoder
        return "K-D"

    @staticmethod
    def get_human_size(value: int) -> str:
        if value < 1_000:
            return str(value)

        return f"{value // 1_000}k"

    @staticmethod
    def get_table_header() -> List[str]:
        return [
            str(label)
            for label in [
                ModelParameterLabel.architecture_type,
                ModelParameterLabel.number_parameters,
                ModelParameterLabel.vocab_size,
                ModelParameterLabel.sequence_length,
                ModelParameterLabel.number_layers,
                ModelParameterLabel.hidden_size,
                ModelParameterLabel.feedforward_hidden_size,
                ModelParameterLabel.number_attention_heads,
                ModelParameterLabel.attention_size,
            ]
        ]
