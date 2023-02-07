from typing import Union, List

import torch

from bgnlp.models.bert import BgPosBert
from bgnlp.lib.utils import IDX2POS


class PosAnalyzer:
    # Training Accuracy: 0.9720
    # Validation Accuracy: 0.9355

    def __init__(self, config, vocab, tokenizer):
        self.config = config
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.model = self._get_model()
    
    def __call__(self, words: Union[List[str], str]):
        pos_result = []

        # When the input words are not in a list, but are a string.
        if isinstance(words, str):
            words = self.tokenizer(words, split_type="word")

        self.model.eval()

        with torch.no_grad():
            for word in words:
                word = word
                tokens = self.tokenizer(word, split_type="symbol")
                tokens = self._add_special_tokens(tokens)

                x = [self.vocab[token] for token in tokens]
                x = torch.LongTensor(x).unsqueeze(0)

                prediction = self.model(x).argmax(-1)
                prediction = IDX2POS[int(prediction.squeeze(0))]

                pos_result.append({
                    "word": word,
                    "pos": prediction
                })
            
        self.model.train()

        return pos_result

    def _get_model(self):
        model = BgPosBert(vocab=self.vocab, n_classes=len(IDX2POS))
        model.load_state_dict(torch.load(self.config.model_path))
        
        return model
    
    def _add_special_tokens(self, tokens):
        tokens = ["[START]"] + tokens + ["[END]"]

        if len(tokens) < self.config.max_size:
            tokens = tokens + ["[PAD]"] * (self.config.max_size - len(tokens))
        else:
            tokens = tokens[:self.config.max_size]

        return tokens
