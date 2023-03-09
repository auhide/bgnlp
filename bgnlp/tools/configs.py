import os
import sys
from typing import Dict
from dataclasses import dataclass, field

from bgnlp.tools.mappings import POS_ID2LABEL, POS_LABEL2ID


PACKAGE_DIR = os.path.dirname(sys.modules["bgnlp"].__file__)


@dataclass
class ModelConfig:
    base_model_id: str
    model_path: str
    device: str = "cpu"


@dataclass
class PosTaggerConfig(ModelConfig):
    base_model_id: str = "rmihaylov/bert-base-bg"
    label2id: Dict[str, int] = field(default_factory=lambda: POS_LABEL2ID)
    id2label: Dict[int, str] = field(default_factory=lambda: POS_ID2LABEL)

    device: str = "cpu"
    model_path: str = os.path.join(
        PACKAGE_DIR, "serialized", "models", "lemma-bert-latest.pt"
    )
