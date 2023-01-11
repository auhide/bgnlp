import os

import pandas as pd
import torch
from torchtext.vocab import build_vocab_from_iterator

from bg_nlp.tokenizers import DefaultTokenizer
from bg_nlp.preprocessing import iterate_corpus


SPECIAL_TOKENS = ["[START]", "[END]", "[PAD]", "[UNK]"]


bg_tokenizer = DefaultTokenizer()

df = pd.read_csv(os.path.join("..", "datasets", "bg-pos", "bg-pos.csv"), sep="\t")
all_word_tokens = [
    symbol 
    for word in df["word"] 
    for symbol in bg_tokenizer(word, split_type="symbol")
]
all_lemma_tokens = [
    symbol 
    for word in df["word"] 
    for symbol in bg_tokenizer(word, split_type="symbol")
]
all_tokens = all_word_tokens + all_lemma_tokens

print("Creating the vocabulary...")
vocab = build_vocab_from_iterator(
    iterate_corpus(tokens=all_tokens, batch_size=100),
    specials=SPECIAL_TOKENS
)
vocab.set_default_index(vocab["[UNK]"])
print(f"Vocabulary size: {len(vocab)}")
print("Saving the vocabulary...")
torch.save(vocab, os.path.join(".", "bg_nlp", "vocabs", "symbs_vocab.pt"))
print("Vocabulary saved!")
