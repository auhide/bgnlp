# **bgnlp**: Model-first approach to Bulgarian NLP
<a href="https://colab.research.google.com/drive/1etvcxad0f754pjSdjremDftq16o_oMTh?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

[![Downloads](https://static.pepy.tech/personalized-badge/bgnlp?period=total&units=international_system&left_color=grey&right_color=blue&left_text=pip%20downloads)](https://pypi.org/project/bgnlp/)

```sh
pip install bgnlp
```

## Package functionalities
- [Part-of-speech](#pos)
- [Lemmatization](#lemma)
- [Named Entity Recognition](#ner)
- [Keyword Extraction](#keywords)
- [Commatization](#comma)

> Please note - only the first time you run one of these operations a model will be downloaded! Therefore, the first run might take more time.


<a id="pos"></a>

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

<a id="lemma"></a>

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

<a id="ner"></a>

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


<a id="keywords"></a>

### Keyword Extraction
```python
from bgnlp import extract_keywords


# Reading the text from a file, since it may be large, hence it wouldn't be 
# pleasant to write it directly here.
# The current input is this Bulgarian news article (only the text, no HTML!):
# https://novini.bg/sviat/eu/781622
with open("input_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Extracting keywords with probability of at least 0.5.
keywords = extract_keywords(text, threshold=0.5)
print("Keywords:")
pprint(keywords)
```
```bash
Keywords:
[{'keyword': 'Еманюел Макрон', 'score': 0.8759163320064545},
 {'keyword': 'Г-7', 'score': 0.5938143730163574},
 {'keyword': 'Япония', 'score': 0.607077419757843}]
```

<a id="comma"></a>

### Commatization
```python
from pprint import pprint

from bgnlp import commatize


text = "Човекът искащ безгрижно писане ме помоли да създам този модел."

print("Without metadata:")
print(commatize(text))

print("\nWith metadata:")
pprint(commatize(text, return_metadata=True))
```
```bash
Without metadata:
Човекът, искащ безгрижно писане, ме помоли да създам този модел.

With metadata:
('Човекът, искащ безгрижно писане, ме помоли да създам този модел.',
 [{'end': 12,
   'score': 0.9301406145095825,
   'start': 0,
   'substring': 'Човекът, иск'},
  {'end': 34,
   'score': 0.93571537733078,
   'start': 24,
   'substring': ' писане, м'}])
```