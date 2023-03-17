import re

from torchtext.vocab import Vocab


class CharacterBasedPreTokenizer:
    
    def __call__(self, text: str):
        text = re.sub(r"([,\.\/!?;:\'\"])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        
        return list(text)


class CharacterBasedTokenizer:
    
    def __init__(self, pretokenizer: CharacterBasedPreTokenizer, vocab: Vocab):
        self.pretokenizer = pretokenizer
        # Mapping the tokens to IDs using vocab.
        # Inversely, we can use self.vocab.lookup_token(8) to map
        # the IDs to their respective tokens.
        self.vocab = vocab
        
    def __call__(
        self, 
        word, pos, 
        index=True, pad=True, truncate=False, max_length=50,
        cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]"
    ):
        tokens = [cls_token] + self.pretokenizer(word) + [sep_token] + [pos] + [sep_token]

        if pad:
            if max_length > len(tokens):
                padding = [pad_token] * (max_length - len(tokens))
                tokens += padding
            
            if truncate:
                if max_length <= len(tokens):
                    tokens = tokens[:max_length]
        
        # Adding attention mask, which masks the padding.
        attention_mask = [1 if token != pad_token else 0 for token in tokens]

        if index:
            return [self.vocab[token] for token in tokens], attention_mask

        return tokens, attention_mask
    
    def __len__(self):
        return len(self.vocab)
