from enum import Enum


class ModelParameterLabel(str, Enum):
    name = "Nazwa z repozytorium HuggingFace Models"
    human_name = "Nazwa modelu"
    architecture_type = "Rodzaj architektury"
    number_parameters = "Liczba parametrów"
    number_layers = "Liczba warstw (K/D)"
    hidden_size = "Wymiar modelu (K/D)"
    feedforward_hidden_size = "Wymiar liniowej transformacji (K/D)"
    number_attention_heads = "Liczba głowic atencji (K/D)"
    attention_size = "Wymiar głowicy atencji (K/D)"
    sequence_length = "Długość sekwencji"
    vocab_size = "Wielkość słownika"

    def __str__(self) -> str:
        return self.value
