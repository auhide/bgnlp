import os
import sys
from typing import Dict
from dataclasses import dataclass, field

from bgnlp.tools.mappings import POS_ID2LABEL, POS_LABEL2ID


PACKAGE_DIR = os.path.dirname(sys.modules["bgnlp"].__file__)


@dataclass
class ModelConfig:
    model_path: str
    device: str = "cpu"


@dataclass
class PosTaggerConfig(ModelConfig):
    device: str = "cpu"
    model_url: str = "https://drive.google.com/uc?id=1MHADXPM5oYRMz4nyPqaKeW2zQSctOxBQ"
    model_path: str = os.path.join(
        PACKAGE_DIR, "serialized", "models", "pos-bert.pt"
    )

    base_model_id: str = "rmihaylov/bert-base-bg"
    label2id: Dict[str, int] = field(default_factory=lambda: POS_LABEL2ID)
    id2label: Dict[int, str] = field(default_factory=lambda: POS_ID2LABEL)


@dataclass
class LemmaTaggerConfig(ModelConfig):
    device: str = "cpu"
    model_url: str = "https://drive.google.com/uc?id=1U0Tb1AN0Rzdz_WsjMayEKK1rHuyZWdfK"
    model_path: str = os.path.join(
        PACKAGE_DIR, "serialized", "models", "lemma-bert.pt"
    )

    vocab_path: str = os.path.join(
        PACKAGE_DIR, "serialized", "vocabs", "cb-vocab.pt"
    )


@dataclass
class NerTaggerConfig(ModelConfig):
    device: str = "cpu"
    # Since I have uploaded this model to Huggingface's Model Hub, it is being 
    # downloaded only using this model ID:
    model_path: str = "auhide/bert-bg-ner"
