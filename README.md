# **bgnlp**: Model-first approach to Bulgarian NLP
<a href="https://colab.research.google.com/drive/1etvcxad0f754pjSdjremDftq16o_oMTh?usp=sharing"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>

[![Downloads](https://static.pepy.tech/personalized-badge/bgnlp?period=total&units=international_system&left_color=grey&right_color=blue&left_text=pip%20downloads)](https://pypi.org/project/bgnlp/)

```sh
pip install bgnlp
```

## Package functionalities

### Part-of-speech (PoS) tagging

```python
from bgnlp import pos


print(pos("Това е библиотека за обработка на естествен език."))
```

```json
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
```

### Lemmatization

```python
from bgnlp import lemmatize


text = "Добре дошли!"
print(lemmatize(text))
```

```bash
[{'word': 'Добре', 'lemma': 'Добре'}, {'word': 'дошли', 'lemma': 'дойда'}, {'word': '!', 'lemma': '!'}]
```

```python
# Generating a string of lemmas.
print(lemmatize(text, as_string=True))
```

```bash
Добре дойда!
```

### Named Entity Recognition (NER) tagging

Currently, the available NER tags are:
- `PER` - Person
- `ORG` - Organization
- `LOC` - Location

```python
from bgnlp import ner


text = "Барух Спиноза е роден в Амстердам"

print(f"Input: {text}")
print("Result:", ner(text))
```

```bash
Input: Барух Спиноза е роден в Амстердам
Result: [{'word': 'Барух Спиноза', 'entity_group': 'PER'}, {'word': 'Амстердам', 'entity_group': 'LOC'}]
```


### Using a Config object
A tagger Config is used to define the underlying model. 

You can change the device on which it makes inference:
```python
# Make inference using the GPU (by default it is "cpu"):
config = NerTaggerConfig(device="cuda")
ner = NerTagger(config=config)
# ...
```

You can also change the path to the model weights. For `NerTagger` you can directly pass the HuggingFace's Model Hub path. All other taggers use weights uploaded to Google Drive.
```python
# Define the path to the model weights. It can be a single .pt file or a path to HuggingFace's Model Hub (only for NerTagger).
config = NerTaggerConfig(model_path="path/to/model")
ner = NerTagger(config=config)
# ...
```
Please, note that the model should be of the same architecture as the one used by the certain Tagger.
