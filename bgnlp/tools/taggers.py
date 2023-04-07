"""
Natural language Taggers for Bulgarian. They are all model-based.
"""
import os
import re
from typing import List, Dict
from abc import ABC, abstractmethod

import torch
from torch import nn
import gdown
from transformers import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification

from bgnlp.models import LemmaBert
from bgnlp.tools.mixins import SubwordMixin
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


class PosTagger(BaseTagger, SubwordMixin):
    """Part-of-speech tagger. Tagging is done using a BERT model trained on 
    [Wiki1000+ Bulgarian corpus](http://dcl.bas.bg/wikiCorpus.html).
    
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
        """Tag each one of the words in `text` with a part-of-speech tag.

        Args:
            text (str): Text in Bulgarian.
            max_len (int, optional): The maximum number of words that you can have in `text`. Defaults to 64.

        Returns:
            List[Dict[str, str]]: List of dictionaries for each word and its tags.

        Example::

            >>> from bgnlp import PosTagger, PosTaggerConfig
            >>> config = PosTaggerConfig()
            >>> pos = PosTagger(config=config)
            >>> pos("Това е библиотека за обработка на естествен език.")
            [{
                "word": "Това",
                "tag": "PDOsn",
                "bg_desc": "местоимение",
                "en_desc": "pronoun"
            }, {
                "word": "е",
                "tag": "VLINr3s",
                "bg_desc": "глагол",
                "en_desc": "verb"
            }, {
                "word": "библиотека",
                "tag": "NCFsof",
                "bg_desc": "съществително име",
                "en_desc": "noun"
            }, {
                "word": "за",
                "tag": "R",
                "bg_desc": "предлог",
                "en_desc": "preposition"
            }, {
                "word": "обработка",
                "tag": "NCFsof",
                "bg_desc": "съществително име",
                "en_desc": "noun"
            }, {
                "word": "на",
                "tag": "R",
                "bg_desc": "предлог",
                "en_desc": "preposition"
            }, {
                "word": "естествен",
                "tag": "Asmo",
                "bg_desc": "прилагателно име",
                "en_desc": "adjective"
            }, {
                "word": "език",
                "tag": "NCMsom",
                "bg_desc": "съществително име",
                "en_desc": "noun"
            }, {
                "word": ".",
                "tag": "U",
                "bg_desc": "препинателен знак",
                "en_desc": "punctuation"
            }]
        """
        self.max_len = max_len

        return self.predict(text)

    def predict(self, text: str) -> List[Dict[str, str]]:
        """Tag each one of the words in `text` with a part-of-speech tag.

        Args:
            text (str): Text in Bulgarian.

        Returns:
            List[Dict[str, str]]: List of dictionaries for each word and its tags.
        """
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
        """Get the tokenizer for the used model.

        Returns:
            AutoTokenizer: HuggingFace AutoTokenizer.
        """
        return AutoTokenizer.from_pretrained(self.config.base_model_id)

    def get_model(self) -> nn.Module:
        """Get the model used for tagging. When this method is called for the first
        time, the model is downloaded. Afterwards, it should be part of your package.

        Returns:
            nn.Module: PyTorch Module.
        """
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
        """Prepare the string `text` for the model. 
        
        This includes:
        - Surrounding punctuation with whitespace
        - Converting multiple consecutive whitespaces into one.

        Args:
            text (str): _description_

        Returns:
            str: _description_
        """
        text = re.sub(r"([.,!?:;])+", r" \1 ", text)
        text = re.sub(r"(\s+)", " ", text)

        return text.strip()
    
    def _format_prediction(self, input_tokens: List[str], prediction: List[str]) -> List[Dict[str, str]]:
        """Format the prediction returned from the model.

        Since the tokenizer of the model is a subword tokenizer, the words
        are split into multiple subwords. The task of this method is to merge 
        them and then create a dictionaries with their tags.

        Args:
            input_tokens (List[str]): Input words (tokens) as strings.
            prediction (List[str]): Predicted tags.

        Returns:
            List[Dict[str, str]]: All found words and their tags.
        """
        tags = []

        # This method is taken from the SubwordMixin class.
        # Here, I am passing all tokens except for the 1st one - [CLS].
        tokens = self.subwords_to_words(input_tokens[1:])

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
        """Map `tag` to its description based on `lang`

        Args:
            lang (str): Language - either 'bg' or 'en'.
            tag (str): PoS tag as a string.

        Returns:
            str: The hardcoded tag description, based on `self.TAGS_MAPPING`.
        """
        first_tag = tag[0]
        description = self.TAGS_MAPPING[first_tag][lang]

        return description


class Lemmatizer(BaseTagger):
    """Lemmatize a word. This is only for single-word lemmatization. If you want 
    to lemmatize multiple words, please use :ref:`LemmaTagger`.

    Args:
        config (ModelConfig): Configuration of the :ref:`LemmaBert` model.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()

    def __call__(self, word: str, pos: str) -> str:
        """Convert `word` into its lemma.

        Args:
            word (str): Word in Bulgarian.
            pos (str): Its part-of-speech tag.

        Returns:
            str: The lemma of `word`.

        Example::
            >>> from bgnlp import LemmaTaggerConfig
            >>> from bgnlp.tools.taggers import Lemmatizer
            >>> lemma = Lemmatizer(config=LemmaTaggerConfig())
            >>> lemma("езикът", "Ns")
            език
        """
        return self.predict(word, pos)

    def get_tokenizer(self) -> CharacterBasedTokenizer:
        """Get the tokenizer used by `LemmaBert`. It is a character-based one.

        Returns:
            CharacterBasedTokenizer: The tokenizer.
        """
        vocab = torch.load(self.config.vocab_path)
        pretokenizer = CharacterBasedPreTokenizer()

        tokenizer = CharacterBasedTokenizer(
            pretokenizer=pretokenizer,
            vocab=vocab
        )

        return tokenizer

    def get_model(self) -> nn.Module:
        """Get the `LemmaBert` model.

        Returns:
            nn.Module: PyTorch Module.
        """
        bert = LemmaBert(
            vocab_size=len(self.tokenizer), 
            output_size=len(self.tokenizer),
            device=self.config.device
        ).to(self.config.device)
        bert = self.load_model(bert)

        return bert

    def predict(self, word: str, pos: str) -> str:
        """Convert `word` into its lemma.

        Args:
            word (str): Word in Bulgarian.
            pos (str): Its part-of-speech tag.

        Returns:
            str: The lemma of `word`.
        """
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
    """Find the lemmas of a string with one or more words.

    Args:
        config (ModelConfig): Configuration of the :ref:`LemmaBert` model.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def __call__(self, text: str, as_string: bool = False, additional_info: bool = False) -> List[Dict[str, str]]:
        """Find the lemmas of `text`. `text` should preferably be a semantically correct sentence or sentences, since the lemma sometimes changes based on context.

        Args:
            text (str): String with one or more Bulgarian words.
            as_string(str): Whether the lemmatization result should be a string or a dictionary.
            additional_info (bool, optional): Whether the output should constist of more data about each word (mainly PoS information). Defaults to False.

        Returns:
            List[Dict[str, str]]: List of dictionaries. Each dictionary has a word and a `lemma` key with a value - its lemma. If `additional_info`=True, the dictionary has PoS data.

        Example::
            >>> from bgnlp import LemmaTaggerConfig, LemmaTagger
            >>> lemma = LemmaTagger(config=LemmaTaggerConfig())
            >>> text = "Добре дошли!"
            >>> print("Input:", text)
            >>> print("Output:", lemma(text))
            [{'word': 'Добре', 'lemma': 'Добре'}, {'word': 'дошли', 'lemma': 'дойда'}, {'word': '!', 'lemma': '!'}]

            >>> lemma = LemmaTagger(config=LemmaTaggerConfig())
            >>> text = "Добре дошли!"
            >>> print("Input:", text)
            >>> print("Output:", lemma(text, as_string=True))
            Input: Добре дошли!
            Output: Добре дойда!
        """
        self.additional_info = additional_info

        pos = PosTagger(config=PosTaggerConfig())
        lemma = Lemmatizer(config=self.config)

        if as_string:
            return self._str_predict(
                text=text,
                pos_model=pos,
                lemma_model=lemma
            )
        
        return self._dict_predict(
            text=text, 
            pos_model=pos,
            lemma_model=lemma
        )

    def _dict_predict(self, text: str, pos_model: PosTagger, lemma_model: Lemmatizer) -> List[Dict[str, str]]:
        """Find the lemmas of each word in `text`. Then, return a dictionary
        with each word and its lemma.

        Args:
            text (str): Bulgarian text.
            pos_model (PosTagger): Part-of-Speech model.
            lemma_model (Lemmatizer): Lemmatization model.

        Returns:
            List[Dict[str, str]]: List of dictionaries. Each dictionary has a word and its lemma.
        """
        result = []

        for pos_result in pos_model(text):
            pos_result["lemma"] = lemma_model(
                word=pos_result["word"],
                pos=pos_result["tag"]
            )
            if self.additional_info:
                result.append(pos_result)
            else:
                result.append({
                    "word": pos_result["word"],
                    "lemma": pos_result["lemma"]
                })

        return result

    def _str_predict(self, text: str, pos_model: PosTagger, lemma_model: Lemmatizer) -> str:
        """Find the lemmas of `text` and return a string with them.

        Args:
            text (str): Bulgarian text.
            pos_model (PosTagger): Part-of-speech model.
            lemma_model (Lemmatizer): Lemmatization model.

        Returns:
            str: String with the lemmas of `text`.
        """
        result = []

        for pos_result in pos_model(text):
            pos_result["lemma"] = lemma_model(
                word=pos_result["word"],
                pos=pos_result["tag"]
            )

            result.append(pos_result["lemma"])

        result = " ".join(result)
        # Removing the left whitespace around punctuation.
        result = re.sub(r"\s([,.\?\!\:\;]+)\s?", r"\1 ", result)

        return result


class NerTagger(BaseTagger, SubwordMixin):
    """Named Entity Recognition (NER) tagging for Bulgarian text.

    Args:
        config (ModelConfig): Configuration of the NerTagger.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()

    def __call__(self, text: str) -> List[Dict[str, str]]:
        """Find entities in `text`. These entities may be:
        - `PER` - Person
        - `ORG` - Organization
        - `LOC` - Location

        Args:
            text (str): String of Bulgarian words.

        Returns:
            List[Dict[str, str]]: List of dictionaries. Each dictionary has a word and its NER tag.

        Example::
            >>> from bgnlp import NerTagger, NerTaggerConfig


            >>> ner = NerTagger(config=NerTaggerConfig())
            >>> text = "Барух Спиноза е роден в Амстердам"

            >>> print(f"Input: {text}")
            >>> print("Result:", ner(text))
            Input: Барух Спиноза е роден в Амстердам
            Result: [{'word': 'Барух Спиноза', 'entity_group': 'PER'}, {'word': 'Амстердам', 'entity_group': 'LOC'}]

        """
        return self.predict(text)

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.model_path)

    def get_model(self):
        return AutoModelForTokenClassification.from_pretrained(self.config.model_path)

    def predict(
        self, 
        text: str, 
        label2id={
            0: "O",
            1: "B-PER", 2: "I-PER", 
            3: "B-ORG", 4: "I-ORG", 
            5: "B-LOC", 6: "I-LOC"
        }
    ) -> List[Dict[str, str]]:
        tokens_data = self.tokenizer(text)
        tokens = self.tokenizer.convert_ids_to_tokens(tokens_data["input_ids"])
        words = self.subwords_to_words(tokens)

        input_ids = torch.LongTensor(tokens_data["input_ids"]).unsqueeze(0)
        attention_mask = torch.LongTensor(tokens_data["attention_mask"]).unsqueeze(0)

        out = self.model(input_ids, attention_mask=attention_mask).logits
        out = out.argmax(-1).squeeze(0).tolist()

        prediction = [label2id[idx] if idx in label2id else idx for idx in out]

        return self._merge_words_and_predictions(
            words=words, entities=prediction
        )
    
    def _merge_words_and_predictions(self, words: List[str], entities: List[str]) -> List[Dict[str, str]]:
        result = []
        curr_word = []

        for i, (word, entity) in enumerate(zip(words[1:], entities[1:])):
            if "B-" in entity:
                if curr_word:
                    curr_word = " ".join(curr_word)
                    result.append({
                        "word": curr_word,
                        "entity_group": entities[i][2:]
                    })
                    curr_word = [word]
                else:
                    curr_word.append(word)

            if "I-" in entity:
                curr_word.append(word)
            
            if "O" == entity:
                if curr_word:
                    curr_word = " ".join(curr_word)
                    result.append({
                        "word": curr_word,
                        "entity_group": entities[i][2:]
                    })
                
                curr_word = []

        return result
