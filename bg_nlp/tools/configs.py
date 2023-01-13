import os
from dataclasses import dataclass


class Config:
    l_rate: float
    batch_size: float
    epochs: int
    device: str
    model_path: str


@dataclass
class BgLemmatizerConfig(Config):
    # Standart parameters:
    device: str
    l_rate: float = 1e-3
    batch_size: float = 256
    epochs: int = 4
    model_path: str = os.path.join(
        ".", "bg_nlp", "serialized", "models", "lemmatizer.pt"
    )

    # Model parameters:
    enc_embed_size: int = 256
    dec_embed_size: int = 256
    enc_hidden_size: int = 256
    dec_hidden_size: int = 256
    enc_dropout: float = 0.5
    dec_dropout: float = 0.5
    # I've chosen these values since the maximum length of a word was 13 and I've
    # counted the [START] and [END] tokens, hence the max. sizes are 15.
    max_word_size: int = 15
    max_lemma_size: int = 15
