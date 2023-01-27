from typing import List

import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab

from bgnlp.tools.tokenizers import DefaultPreTokenizer


class LemmatizationDataset(Dataset):
    
    def __init__(
        self, 
        words: List[str], lemmas: List[str], 
        tokenizer: DefaultPreTokenizer, vocab: Vocab,
        words_max_size: int, lemmas_max_size: int,
        padding_token="[PAD]", start_token="[START]", end_token="[END]"
    ):
        self.words = words
        self.lemmas = lemmas
        self.lemmas_are_passed = True

        # Create default target sequences, when lemmas is None.
        # That's going to be used for the creation of a prediction dataset.
        if lemmas is None:
            self.lemmas_are_passed = False
            default_target = " ".join(
                ["[START]"] + ["[PAD]"] * (lemmas_max_size - 1)
            )
            self.lemmas = [default_target for _ in range(len(words))]

        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_size_1 = words_max_size
        self.max_size_2 = lemmas_max_size

        self.padding = padding_token
        self.start_token = start_token
        self.end_token = end_token

        self.x, self.y = self._get_inouts()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

    def _get_inouts(self):
        x, y = [], []
        
        for word, lemma in zip(self.words, self.lemmas):
            word_tokens = self.tokenizer(word, split_type="symbol")
            if self.lemmas:
                lemma_tokens = self.tokenizer(lemma, split_type="symbol")
            else:
                lemma_tokens = self.tokenizer(lemma, split_type="word")

            # Adding start and end tokens.
            word_tokens = [self.start_token] + word_tokens + [self.end_token]
            lemma_tokens = [self.start_token] + lemma_tokens + [self.end_token]

            # Add padding to the lists of tokens.
            word_tokens = self._pad_sequence(word_tokens, self.max_size_1)
            lemma_tokens = self._pad_sequence(lemma_tokens, self.max_size_2)

            x.append([self.vocab[symbol] for symbol in word_tokens])
            y.append([self.vocab[symbol] for symbol in lemma_tokens])
        
        return torch.LongTensor(x), torch.LongTensor(y)

    def _pad_sequence(self, sequence, max_size):
        # When we must pad the sequence.
        if max_size > len(sequence):
            return sequence + [self.padding] * (max_size - len(sequence))
        
        # When we must slice the sequence.
        return sequence[:max_size]
