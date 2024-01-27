"""
Natural language Taggers for Bulgarian. They are all model-based.
"""
import os
import re
from typing import List, Dict, Any, Tuple, Union
from abc import ABC, abstractmethod

import torch
from torch import nn
import gdown
from transformers import logging
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

from bgnlp.models import LemmaBert
from bgnlp.tools.mixins import SubwordMixin
from bgnlp.tools.tokenizers import (
    CharacterBasedPreTokenizer, CharacterBasedTokenizer
)
from bgnlp.tools.configs import (
    ModelConfig, 
    PosTaggerConfig,
)


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
            model_obj.load_state_dict(
                torch.load(self.config.model_path, map_location=self.config.device), 
                strict=False
            )
        else:
            # Downloading the model if it doesn't exist locally.
            # The model is not deployed with the PyPI package - hence, the download below.
            gdown.download(
                self.config.model_url, 
                self.config.model_path, 
                quiet=False
            )
            model_obj.load_state_dict(
                torch.load(self.config.model_path, map_location=self.config.device), 
                strict=False
            )

        return model_obj


class PosTagger(BaseTagger, SubwordMixin):
    """Part-of-speech tagger. Tagging is done using a BERT model trained on 
    [Wiki1000+ Bulgarian corpus](http://dcl.bas.bg/wikiCorpus.html).
    
    Args:
        config (ModelConfig): Configuration of the PosTagger.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

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

            >>> from bgnlp import pos
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
        self.model = self.get_model().to(self.config.device)
        self.tokenizer = self.get_tokenizer()

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
        self.model = self.get_model()
        self.tokenizer = self.get_tokenizer()

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
            >>> from bgnlp import lemmatize
            >>> text = "Добре дошли!"
            >>> # Return the lemmas as a dictionary.
            >>> print("Input:", text)
            >>> print("Output:", lemmatize(text))
            [{'word': 'Добре', 'lemma': 'Добре'}, {'word': 'дошли', 'lemma': 'дойда'}, {'word': '!', 'lemma': '!'}]

            >>> # Or return the lemmas as a string.
            >>> print("Output:", lemmatize(text, as_string=True))
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
            >>> from bgnlp import ner
            >>> text = "Барух Спиноза е роден в Амстердам"
            >>> print(f"Input: {text}")
            >>> print("Result:", ner(text))
            Input: Барух Спиноза е роден в Амстердам
            Result: [{'word': 'Барух Спиноза', 'entity_group': 'PER'}, {'word': 'Амстердам', 'entity_group': 'LOC'}]

        """
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()

        text = self._preprocess_text(text)
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

        for i, (word, entity) in enumerate(zip(words[1:], entities)):
            if "B-" in entity:
                if curr_word:
                    curr_word = " ".join(curr_word)
                    result.append({
                        "word": curr_word,
                        "entity_group": entities[i - 1][2:]
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
                        "word": self._remove_punctuation(curr_word),
                        "entity_group": entities[i - 1][2:]
                    })
                
                curr_word = []

        return result
    
    def _preprocess_text(self, text: str) -> str:
        # Remove the whitespace before punctuation.
        text = re.sub(r"\s+([,\.\?!;:\'\"\(\)\[\]„”])", r"\1", text)
        # Leave out only a single whitespace.
        text = re.sub(r"\s+", " ", text)
        
        return text

    def _remove_punctuation(self, text: str) -> str:
        return re.sub(r"([,\.\?!;:\'\"\(\)\[\]„”])", "", text)


class KeywordsTagger(BaseTagger):
    """Keyword Extraction tagger for Bulgarian texts.

    Args:
        config (ModelConfig): The model configuration.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def __call__(self, text: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Extract keywords from Bulgarian texts.

        Args:
            text (str): The source text from which you are going to extract.
            threshold (float, optional): Threshold based on which some of the keywords with lower probabilties might be excluded. Defaults to 0.5.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries describing each keyword in `text`.

        Example::
            >>> from bgnlp import extract_keywords
            >>> with open("input_text.txt", "r", encoding="utf-8") as f:
            >>>     text = f.read()
            >>> # Here threshold is optional, it defaults to 0.5.
            >>> extract_keywords(text, threshold=0.6)
            [{'keyword': 'Еманюел Макрон', 'score': 0.8759163320064545},
            {'keyword': 'Г-7', 'score': 0.5938143730163574},
            {'keyword': 'Япония', 'score': 0.607077419757843}]
        """
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()

        return self.predict(text, threshold)

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.model_path)

    def get_model(self):
        return AutoModelForTokenClassification.from_pretrained(self.config.model_path)

    def predict(self, text: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Extract keywords from Bulgarian texts.

        Args:
            text (str): The source text from which you are going to extract.
            threshold (float, optional): Threshold based on which some of the keywords with lower probabilties might be excluded. Defaults to 0.5.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries describing each keyword in `text`.

        Example::
            >>> from bgnlp import extract_keywords
            >>> with open("input_text.txt", "r", encoding="utf-8") as f:
            >>>     text = f.read()
            >>> # Here threshold is optional, it defaults to 0.5.
            >>> extract_keywords(text, threshold=0.6)
            [{'keyword': 'Еманюел Макрон', 'score': 0.8759163320064545},
            {'keyword': 'Г-7', 'score': 0.5938143730163574},
            {'keyword': 'Япония', 'score': 0.607077419757843}]
        """
        keywords = self._extract_keywords(text, threshold=threshold)

        return self._format_keywords(keywords)
    
    def _format_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mainly responsible for the merging of subkeywords into keywords, i.e. when
        the keyword consists of multiple words - 'Адам Фаузи', the two subkeywords 'Адам' and 'Фаузи'
        are merged into one. This method also merges the probabilities by calculating
        their average.

        Args:
            keywords (List[Dict[str, Any]]): Keywords with their `entity_group` and probability `score`.

        Returns:
            List[Dict[str, Any]]: Merged keywords (in some cases) with their probability scores.
        """
        formatted_keywords = []
        # This is used for keywords that have multiple words.
        current_keywords = []
        scores = []

        for i, kw in enumerate(keywords):
            if kw["entity_group"] == "B-KWD":
                if i > 0:
                    formatted_keywords.append({
                        "keyword": " ".join(current_keywords),
                        # Calculating the average score of all keywords in `current_keywords`.
                        "score": sum(scores) / len(scores)
                    })
                current_keywords = []
                scores = []
                current_keywords.append(kw["entity"])
                scores.append(kw["score"])

            if kw["entity_group"] == "I-KWD":
                current_keywords.append(kw["entity"])
                scores.append(kw["score"])

            # When the last keyword is of any type - it should be added to 
            # `formatted_keywords`.
            if i == len(keywords) - 1:
                formatted_keywords.append({
                    "keyword": " ".join(current_keywords),
                    # Calculating the average score of all keywords in `current_keywords`.
                    "score": sum(scores) / len(scores)
                })

        return formatted_keywords

    def _extract_keywords(
        self,
        text: str,
        max_len: int = 300,
        id2group = {
            # Indicates that this is not a keyword.
            0: "O",
            # Begining of keyword.
            1: "B-KWD",
            # Additional keywords (might also indicate the end of a keyword sequence).
            # You can merge these with the begining keyword `B-KWD`.
            2: "I-KWD",
        },
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Here the text is preprocessed, tokenized and then sent to the model
        for inference. There are comments on each step.

        Args:
            text (str): Raw text.
            max_len (int, optional): Maximum sequence length passed to the tokenizer. Defaults to 300.
            id2group (dict, optional): ID to Group mapping for the entity groups. Defaults to { 0: "O", 1: "B-KWD", 2: "I-KWD", }.
            threshold (float, optional): Threshold based on which some of the keywords with lower probabilties might be excluded. Defaults to 0.5.

        Returns:
            List[Dict[str, Any]]: Each found entity/keyword with its entity group and probability score.
        """
        # Preprocess the text.
        # Surround punctuation with whitespace and convert multiple whitespaces
        # into single ones.
        text = re.sub(r"([,\.?!;:\'\"\(\)\[\]„”])", r" \1 ", text)
        text = re.sub(r"\s+", r" ", text)
        words = text.split()

        # Tokenize the processed `text` (this includes padding or truncation).
        tokens_data = self.tokenizer(
            text.strip(), 
            padding="max_length", 
            max_length=max_len, 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids = tokens_data.input_ids
        attention_mask = tokens_data.attention_mask

        # Predict the keywords.
        out = self.model(input_ids, attention_mask=attention_mask).logits
        # Softmax the last dimension so that the probabilities add up to 1.0.
        out = out.softmax(-1)
        # Based on the probabilities, generate the most probable keywords.
        out_argmax = out.argmax(-1)
        prediction = out_argmax.squeeze(0).tolist()
        probabilities = out.squeeze(0)
        
        return [
            {
                # Since the list of words does not have a [CLS] token, the index `i`
                # is one step forward, which means that if we want to access the 
                # appropriate keyword we should use the index `i - 1`.
                "entity": words[i - 1],
                "entity_group": id2group[idx],
                "score": float(probabilities[i, idx])
            } 
            for i, idx in enumerate(prediction) 
            if (idx == 1 or idx == 2) and float(probabilities[i, idx]) > threshold
        ]


class PunctuationTagger(BaseTagger):

    def __init__(self, config: ModelConfig):
        """Punctuator for Bulgarian texts.

        Args:
            config (ModelConfig): Model configuration. By default it is `bgnlp.tools.configs.PunctuationTaggerConfig`.
        """
        self.config = config

    def __call__(
        self, 
        text: str, 
        threshold: float = 0.5, 
        return_metadata: bool = False
    ) -> Union[str, Tuple[str, List[str]]]:
        """Punctuate Bulgarian texts.

        Args:
            text (str): Text that's going to be punctuated.
            threshold (float, optional): Probability threshold for punctuation. Defaults to 0.5.
            return_metadata (bool, optional): Whether to return metadata like probabilty for each punctuation. Defaults to False.

        Returns:
            Union[str, Tuple[str, List[str]]]: Either a string of the punctuated text, or the punctuated text with its metadata.
        """
        # These two attributes are equal to the same string but I wanted to 
        # comply with the structure of the other classes.
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()

        self.punctuate = pipeline(
            "token-classification", 
            model=self.model, 
            tokenizer=self.tokenizer
        )

        return self.predict(text=text, threshold=threshold, return_metadata=return_metadata)

    def get_model(self) -> str:
        """Returning the HuggingFace model path, because the inference will be made
        using a HF `pipeline`."""
        return self.config.model_path

    def get_tokenizer(self) -> str:
        """Returning the HuggingFace model path, because the inference will be made
        using a HF `pipeline`."""
        return self.config.model_path

    def predict(
        self,
        text: str,
        # TODO: Use this map when there are more tags. Currently, the model infers
        # commas only. 
        punct_map: Dict[str, str] = {
            "B-CMA": ",",
        },
        threshold=0.5,
        return_metadata=False
    ) -> Union[str, Tuple[str, List[str]]]:
        """Tag where the commas should be. Then punctuate the input string `text`.

        Args:
            text (str): Text that's going to be punctuated.
            punct_map (_type_, optional): Punctuation tag map. Defaults to { "B-CMA": ",", }.
            threshold (float, optional): Probability threshold for the punctuation tags. Defaults to 0.5.
            return_metadata (bool, optional): Whether to return the punctuation metadata or not. Defaults to False.

        Returns:
            Union[str, Tuple[str, List[str]]]: _description_
        """
        text = re.sub(",", "", text)

        entities = self.punctuate(text)
        substrings = []
        b_entities_count = 0

        result = text
        
        for i, ent in enumerate(entities):
            if "B-" in ent["entity"] and ent["score"] >= threshold:
                # This is basically <left-tokens><comma> <right-tokens>.
                result = f"{result[:ent['end'] + b_entities_count]}, {result[ent['end'] + 1 + b_entities_count:]}"
                
                # Handling cases in which the right token is not available in the sequence
                # and cannot be caught.
                try:
                    left_token = result[ent['start'] + b_entities_count:ent['end'] + b_entities_count]
                    right_token = result[ent['end'] + 1 + b_entities_count:entities[i + 1]["end"] + b_entities_count]
                except IndexError:
                    continue

                substrings.append({
                    "substring": f"{left_token},{right_token}", 
                    "score": (ent["score"] + entities[i + 1]["score"]) / 2,
                    "start": ent["start"] + b_entities_count,
                    "end": entities[i + 1]["end"] + b_entities_count,
                })

                b_entities_count += 1

        if return_metadata:
            return result, substrings
                    
        return result
