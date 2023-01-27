import os

import pandas as pd

from bgnlp.tools.vocabs import VocabBuilder
from bgnlp.tools.tokenizers import DefaultPreTokenizer


df = pd.read_csv(os.path.join("..", "datasets", "bg-pos", "bg-pos.csv"), sep="\t")
# Join the words to create a corpus string.
corpus = []
for word, lemma in zip(df["word"], df["lemma"]):
    corpus.append(f"{word} {lemma}")

corpus = " ".join(corpus)
print("Corpus created!")
SPECIAL_TOKENS = ["[START]", "[END]", "[PAD]", "[UNK]"]


tokenizer = DefaultPreTokenizer()
vocab_builder = VocabBuilder(tokenizer=tokenizer, split_type="symbol")

vocab = vocab_builder.build_vocab(corpus=corpus, special_tokens=SPECIAL_TOKENS)

print("vocab[б]:", vocab["б"])