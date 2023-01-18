import os
import sys

import torch
from torchtext.vocab import Vocab

from bgnlp.lib.exceptions import NonExistantVocab


PACKAGE_DIR = os.path.dirname(sys.modules["bgnlp"].__file__)


VOCABS_MAPPING = {
    "symbols-vocab": os.path.join(PACKAGE_DIR, "serialized", "vocabs", "svocab.pt")
}


def get_vocab(vocab_name: str) -> Vocab:
    vocab_path = VOCABS_MAPPING.get(vocab_name, None)

    if vocab_path is not None:
        return torch.load(vocab_path)

    raise NonExistantVocab(f"Vocabulary with name '{vocab_name}' does not exist! Try with any of these: {list(VOCABS_MAPPING.keys())}")
