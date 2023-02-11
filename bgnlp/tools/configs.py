import os
import sys
from dataclasses import dataclass


PACKAGE_DIR = os.path.dirname(sys.modules["bgnlp"].__file__)


class Config:
    l_rate: float
    batch_size: float
    epochs: int
    device: str
    model_path: str


@dataclass
class BgLemmatizerConfig(Config):
    # Standard parameters:
    device: str
    l_rate: float = 1e-3
    batch_size: float = 256
    epochs: int = 7
    model_path: str = os.path.join(
        PACKAGE_DIR, "serialized", "models", "lemmatizer.pt"
    )

    # Model parameters:
    enc_embed_size: int = 64
    dec_embed_size: int = 64
    enc_hidden_size: int = 64
    dec_hidden_size: int = 64
    enc_dropout: float = 0.3
    dec_dropout: float = 0.3
    # I've chosen these values since the maximum length of a word was 13 and I've
    # counted int the [START] and [END] tokens as well - hence the max. sizes are 15.
    max_word_size: int = 15
    max_lemma_size: int = 15
    
    teacher_forcing_ratio: float = 0.5


@dataclass
class BgPosAnalyzerConfig(Config):
    model_path: str = os.path.join(
        PACKAGE_DIR, "serialized", "models", "bg-pos-roberta.pt"
    )
    max_size: int = 15
