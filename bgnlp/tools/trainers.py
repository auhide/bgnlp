import logging
from typing import List
from abc import ABC, abstractmethod
from dataclasses import fields

import torch
from torchtext.vocab import Vocab
from torch.utils.data import Dataset
from bgnlp.lib.exceptions import WrongConfigFields

from bgnlp.tools.tokenizers import Tokenizer
from bgnlp.lib.datasets import LemmatizationDataset
from bgnlp.lib.preprocessing import train_validation_split
from bgnlp.tools.configs import BgLemmatizerConfig, Config
from bgnlp.models.seq2seq import Attention, Encoder, Decoder, Seq2Seq, Seq2SeqSession


logging.basicConfig(
    format="%(asctime)s|%(levelname)s| %(message)s", 
    level=logging.INFO
)


class Trainer(ABC):
    
    @abstractmethod
    def train(self) -> torch.nn.Module:
        pass

    def validate_config(self, config: Config):
        for config_field in fields(config):
            if config_field.name not in self.EXPECTED_FIELDS:
                raise WrongConfigFields(f"Wrong config fields! BgLemmatizerTrainer expects these fields:\n{self.EXPECTED_FIELDS}")

class BgLemmatizerTrainer(Trainer):
    EXPECTED_FIELDS = [field.name for field in fields(BgLemmatizerConfig)]
    
    def __init__(
        self, 
        dataset: Dataset, 
        tokenizer: Tokenizer, 
        vocab: Vocab
    ) -> torch.nn.Module:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._vocab = vocab

    def train(
        self, 
        config: Config, 
        log_words: List[str] = None, 
        test_size: float = 0.9,
        save_model: bool = True,
        metrics: dict = None
    ):
        # Validating whether the configuration has the correct fields.
        # If they are incorrect - an exception is raised.
        self.validate_config(config)
        # Splitting the dataset into training and validation sets.
        train_dataset, valid_dataset = train_validation_split(
            self._dataset, train_size=test_size
        )

        # Defining the model and its Modules.
        enc = Encoder(
            vocab_size=len(self._vocab),
            embed_size=config.enc_embed_size,
            hidden_size=config.enc_hidden_size,
            padding_idx=self._vocab["[PAD]"],
            dropout=config.enc_dropout
        )

        attn = Attention(
            encoder_hidden_dim=config.enc_hidden_size, 
            decoder_hidden_dim=config.dec_hidden_size
        )
        
        dec = Decoder(
            vocab_size=len(self._vocab),
            embed_size=config.dec_embed_size,
            hidden_size=config.dec_hidden_size,
            padding_idx=self._vocab["[PAD]"],
            attention=attn,
            dropout=config.dec_dropout
        )

        seq2seq = Seq2Seq(encoder=enc, decoder=dec).to(config.device)

        # Defining the optimizer and the loss function.
        optimizer = torch.optim.Adam(params=seq2seq.parameters(), lr=config.l_rate)
        loss = torch.nn.CrossEntropyLoss().to(config.device)

        # If test words are passed, a fixed dataset is created.
        # It will be used for loggin on each iteration.
        if log_words:
            # Getting a fixed dataset, consisting of the words in 'log_words'.
            fixed_dataset = self._generate_fixed_dataset(
                config=config, 
                words_list=log_words
            )

        logging.info("Starting training session...")
        logging.info(f"Device: {config.device}")

        trainer = Seq2SeqSession(
            model=seq2seq, 
            loss=loss, 
            optimizer=optimizer,
            vocab=self._vocab,
            device=config.device
        )

        if log_words:
            seq2seq = trainer.train(
                train_dataset=self._dataset, valid_dataset=self._dataset,
                batch_size=config.batch_size, epochs=config.epochs,
                fixed_input=fixed_dataset.x.to(config.device), 
                teacher_forcing_ratio=config.teacher_forcing_ratio,
                metrics=metrics
            )
        else:
            seq2seq = trainer.train(
                train_dataset=self._dataset, valid_dataset=self._dataset,
                batch_size=config.batch_size, epochs=config.epochs, 
                teacher_forcing_ratio=config.teacher_forcing_ratio,
                metrics=metrics
            )

        if save_model:
            logging.info(f"Saving model at '{config.model_path}'...")
            torch.save(seq2seq.state_dict(), config.model_path)
            logging.info("Model saved successfully!")

    def _generate_fixed_dataset(self, config, words_list):
        # Dataset used for epoch logging.
        # Usually a couple of words for visual measurement of how good actually
        # the model predicts.
        fixed_dataset = LemmatizationDataset(
            words=words_list, lemmas=None,
            tokenizer=self._tokenizer, 
            vocab=self._vocab,
            words_max_size=config.max_word_size, 
            lemmas_max_size=config.max_lemma_size
        )

        return fixed_dataset
