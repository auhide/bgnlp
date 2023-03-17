import os
import re
from typing import List, Dict
from abc import ABC, abstractmethod

import torch
from torch import nn
import gdown
from transformers import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification

from bgnlp.models.bert import LemmaBert
from bgnlp.tools.tokenizers import (
    CharacterBasedPreTokenizer, CharacterBasedTokenizer
)
from bgnlp.tools.configs import LemmaTaggerConfig, ModelConfig, PosTaggerConfig


# Logging only error messages from HuggingFace.
logging.set_verbosity_error()


class BaseTagger(ABC):

    @abstractmethod
    def get_tokenizer(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def load_model(self, model_obj):
        if os.path.exists(self.config.model_path):
            model_obj.load_state_dict(torch.load(self.config.model_path, map_location=self.config.device))
        else:
            # Downloading the model if it doesn't exist locally.
            # The model is not deployed with the PyPI package - hence, the download below.
            gdown.download(
                self.config.model_url, 
                self.config.model_path, 
                quiet=False
            )
            model_obj.load_state_dict(torch.load(self.config.model_path, map_location=self.config.device))

        return model_obj


class PosTagger(BaseTagger):
    """Part-of-speech tagger. Uses a BERT architecture and is trained on the 
    [Wiki1000+ Bulgarian corpus](http://dcl.bas.bg/wikiCorpus.html).

    Usage:
    

    Args:
        config (ModelConfig): Configuration of the PosTagger.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model().to(self.config.device)

        # TODO: Have better descriptions. Figure out the full meanings of the tags.
        self.TAGS_MAPPING = {
            "N": {
                "en": "noun",
                "bg": "съществително име"
            },
            "A": {
                "en": "adjective",
                "bg": "прилагателно име"
            },
            "P": {
                "en": "pronoun",
                "bg": "местоимение"
            },
            "B": {
                "en": "numeral",
                "bg": "числително име"
            },
            "V": {
                "en": "verb",
                "bg": "глагол"
            },
            "D": {
                "en": "adverb",
                "bg": "наречие"
            },
            "C": {
                "en": "conjunction",
                "bg": "съюз"
            },
            "T": {
                "en": "particle",
                "bg": "частица"
            },
            "R": {
                "en": "preposition",
                "bg": "предлог"
            },
            "I": {
                "en": "interjection",
                "bg": "междуметие"
            },
            "U": {
                "en": "punctuation",
                "bg": "препинателен знак"
            }
        }

    def __call__(self, text: str, max_len=64) -> List[Dict[str, str]]:
        self.max_len = max_len

        return self.predict(text)

    def predict(self, text: str) -> List[Dict[str, str]]:
        text = PosTagger._preprocess_text(text)

        tokens_data = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length"
        )

        # Preparing the input to the model.
        input_ids, attn = tokens_data["input_ids"], tokens_data["attention_mask"]
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.config.device)
        attn =  torch.LongTensor(attn).unsqueeze(0).to(self.config.device)

        # Making a prediction.
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input_ids, attention_mask=attn).logits
            pred = pred.argmax(-1).squeeze(0)
        self.model.train()

        pred = [
            self.config.id2label[int(id_)] 
            for id_ in pred
        ]

        return self._format_prediction(
            input_tokens=self.tokenizer.convert_ids_to_tokens(
                tokens_data["input_ids"]
            ), 
            prediction=pred
        )

    def get_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.config.base_model_id)

    def get_model(self) -> nn.Module:
        bert = AutoModelForTokenClassification.from_pretrained(
            self.config.base_model_id, 
            num_labels=len(self.config.label2id), 
            label2id=self.config.label2id
        )
        bert.resize_token_embeddings(len(self.tokenizer))
        bert = self.load_model(bert)

        return bert

    @staticmethod
    def _preprocess_text(text: str):
        text = re.sub(r"([.,!?:;])+", r" \1 ", text)
        text = re.sub(r"(\s+)", " ", text)

        return text.strip()
    
    def _format_prediction(self, input_tokens: List[str], prediction: List[str]) -> List[Dict[str, str]]:
        tokens = []
        curr_token = ""
        tags = []

        # Merging the subwords into words and removing the '▁' infront of the tokens.
        for token in input_tokens[1:]:
            if token == "[SEP]":
                curr_token = curr_token.replace("▁", "")
                tokens.append(curr_token)
                break

            if "▁" in token and curr_token == "":
                curr_token += token

            elif "▁" in token and curr_token != "":
                curr_token = curr_token.replace("▁", "")
                tokens.append(curr_token)
                curr_token = ""
                curr_token += token

            elif "▁" not in token:
                curr_token += token

        # Getting all predicted tokens (except [CLS]) up until [SEP].
        for token in prediction[1:]:
            if token == "[SEP]":
                break
            tags.append(token)

        result = []
        for word, tag in zip(tokens, tags):
            result.append({
                "word": word,
                "tag": tag,
                "bg_desc": self._get_tag_description(lang="bg", tag=tag),
                "en_desc": self._get_tag_description(lang="en", tag=tag)
            })

        return result

    def _get_tag_description(self, lang: str, tag: str) -> str:
        first_tag = tag[0]
        description = self.TAGS_MAPPING[first_tag][lang]

        return description


class Lemmatizer(BaseTagger):

    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()

    def __call__(self, word: str, pos: str) -> str:
        return self.predict(word, pos)

    def get_tokenizer(self) -> CharacterBasedTokenizer:
        vocab = torch.load(self.config.vocab_path)
        pretokenizer = CharacterBasedPreTokenizer()

        tokenizer = CharacterBasedTokenizer(
            pretokenizer=pretokenizer,
            vocab=vocab
        )

        return tokenizer

    def get_model(self) -> nn.Module:
        bert = LemmaBert(
            vocab_size=len(self.tokenizer), 
            output_size=len(self.tokenizer),
            device=self.config.device
        ).to(self.config.device)
        bert = self.load_model(bert)

        return bert

    def predict(self, word: str, pos: str) -> str:
        # Preparing the input.
        tokens, attention_mask = self.tokenizer(word, pos)

        tokens = torch.LongTensor(tokens).unsqueeze(0)
        attention_mask = torch.LongTensor(attention_mask).unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            pred = self.model(tokens, attention_mask=attention_mask).argmax(-1).squeeze(0).tolist()
            pred = "".join(self.tokenizer.vocab.lookup_tokens(pred))
            pred = re.findall(r"\[CLS\](.+?)\[SEP\]", pred)[0]

        self.model.train()

        if word[0].isupper():
            pred = pred.capitalize()

        return pred


class LemmaTagger:

    def __init__(self, config: ModelConfig):
        self.config = config

    def __call__(self, text: str, additional_info: bool = False) -> List[Dict[str, str]]:
        pos = PosTagger(config=PosTaggerConfig())
        lemma = Lemmatizer(config=LemmaTaggerConfig)
        result = []

        for pos_result in pos(text):
            pos_result["lemma"] = lemma(
                word=pos_result["word"],
                pos=pos_result["tag"]
            )
            if additional_info:
                result.append(pos_result)
            else:
                result.append({
                    "word": pos_result["word"],
                    "lemma": pos_result["lemma"]
                })

        return result
