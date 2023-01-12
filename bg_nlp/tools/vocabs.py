import os

import torch
from torchtext.vocab import Vocab

from bg_nlp.lib.exceptions import NonExistantVocab


VOCABS_MAPPING = {
    "symbols-vocab": os.path.join(".", "bg_nlp", "serialized", "vocabs", "svocab.pt")
}


def get_vocab(vocab_name: str) -> Vocab:
    vocab_path = VOCABS_MAPPING.get(vocab_name, None)

    if vocab_path is not None:
        return torch.load(vocab_path)

    raise NonExistantVocab(f"Vocabulary with name '{vocab_name}' does not exist! Try with any of these: {list(VOCABS_MAPPING.keys())}")
