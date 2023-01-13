import re
import random
from typing import List, Union
from abc import ABC, abstractmethod

import torch
import numpy as np
from torchtext.vocab import Vocab

from bg_nlp.tools.configs import Config
from bg_nlp.lib.datasets import LemmatizationDataset
from bg_nlp.tools.tokenizers import DefaultTokenizer
from bg_nlp.models.seq2seq import Encoder, Attention, Decoder, Seq2Seq


# Setting a random seed, since we want consistant predictions!
# I've done it for general python functions, numpy and PyTorch, since
# there were some discrepancies on inference.
SEED = 42
random.seed(SEED)
np.random.seed(42)
torch.backends.cudnn.benchmark = True
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.use_deterministic_algorithms(True)


class NlpTool(ABC):

    @abstractmethod
    def get_model(self):
        pass


class BgLemmatizer:
    
    def __init__(self, config: Config, vocab: Vocab, tokenizer=DefaultTokenizer()):
        self.config = config
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.model = self._get_model()

        # Index to token/string.
        self.itos = {self.vocab[token]: token for token in self.vocab.vocab.itos_}

    def __call__(self, words: Union[List[str], str]):

        # If words is a string we tokenize the string into words.
        if isinstance(words, str):
            words = self.tokenizer(words, split_type="word")

        # Create a dataset out of 'words'.
        dataset = LemmatizationDataset(
            words=words, lemmas=None,
            tokenizer=self.tokenizer, 
            vocab=self.vocab,
            words_max_size=self.config.max_word_size, 
            lemmas_max_size=self.config.max_lemma_size
        )

        # Moving the tensors to the respective device, depending on the config.
        x, y = dataset.x.to(self.config.device), dataset.y.to(self.config.device)

        # Setting the model to be in evaluation mode before making a prediction,
        # since we use Dropout layers.
        self.model.eval()

        # Disabling autograd and making a prediction.
        with torch.no_grad():
            prediction = self.model(x, y)

        # Getting the symbols with the highest probabilites for each sequence.
        prediction = prediction.argmax(-1).tolist()

        return self._parse_prediction(prediction)

    def _get_model(self):
        enc = Encoder(
            vocab_size=len(self.vocab),
            embed_size=self.config.enc_embed_size,
            hidden_size=self.config.enc_hidden_size,
            padding_idx=self.vocab["[PAD]"],
            dropout=self.config.enc_dropout
        )

        attn = Attention(
            encoder_hidden_dim=self.config.enc_hidden_size, 
            decoder_hidden_dim=self.config.dec_hidden_size
        )
        
        dec = Decoder(
            vocab_size=len(self.vocab),
            embed_size=self.config.dec_embed_size,
            hidden_size=self.config.dec_hidden_size,
            padding_idx=self.vocab["[PAD]"],
            attention=attn,
            dropout=self.config.dec_dropout
        )

        seq2seq = Seq2Seq(encoder=enc, decoder=dec).to(self.config.device)
        seq2seq.load_state_dict(torch.load(self.config.model_path))
        
        return seq2seq

    def _parse_prediction(self, prediction):
        # This list will consist of the string tokens (parsing from index to string).
        tokens_prediction = []
        # This list will consist of the final result - a list of words.
        parsed_prediction = []
        
        # Converting indices to tokens.
        for i, seq in enumerate(prediction):
            tokens_prediction.append([])
            for idx in seq:
                symb = self.itos[idx]
                tokens_prediction[i].append(symb)

        # Merging the words back.
        for seq in tokens_prediction:
            # Joining the list of symbols and getting the word inside
            # the [START]...[END] block.
            seq_string = "".join(seq)
            seq_string = re.sub(r"\[START\](.+?)\[END\].+", r"\1", seq_string)

            parsed_prediction.append(seq_string)

        return parsed_prediction
