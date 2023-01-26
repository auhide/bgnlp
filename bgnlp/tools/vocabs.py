import os
import sys

import torch
from torchtext.vocab import Vocab, build_vocab_from_iterator

from bgnlp.lib.exceptions import NonExistantVocab
from bgnlp.lib.preprocessing import iterate_corpus


PACKAGE_DIR = os.path.dirname(sys.modules["bgnlp"].__file__)


VOCABS_MAPPING = {
    "symbols-vocab": os.path.join(PACKAGE_DIR, "serialized", "vocabs", "svocab.pt")
}


def get_vocab(vocab_name: str) -> Vocab:
    vocab_path = VOCABS_MAPPING.get(vocab_name, None)

    if vocab_path is not None:
        return torch.load(vocab_path)

    raise NonExistantVocab(f"Vocabulary with name '{vocab_name}' does not exist! Try with any of these: {list(VOCABS_MAPPING.keys())}")


class VocabBuilder:

    def __init__(self, tokenizer, split_type="word"):
        self.tokenizer = tokenizer
        self.split_type = split_type

    def build_vocab(self, corpus, special_tokens, unk_token="[UNK]", iter_batch_size=100):
        if unk_token not in special_tokens:
            special_tokens.insert(0, unk_token)

        tokens = self.tokenizer(corpus, split_type=self.split_type)

        self.vocab = build_vocab_from_iterator(
            iterate_corpus(tokens=tokens, batch_size=iter_batch_size),
            specials=special_tokens
        )

        self.vocab.set_default_index(self.vocab[unk_token])

        return self.vocab
    
    def save_vocab(self, path):
        torch.save(self.vocab, path)
