# **bgnlp**: Model-first approach to Bulgarian NLP
<a href="https://colab.research.google.com/drive/1etvcxad0f754pjSdjremDftq16o_oMTh?usp=sharing"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>

```sh
pip install bgnlp
```

## Package functionalities

### Part-of-speech tagging

```python
from bgnlp import PosTagger, PosTaggerConfig

config = PosTaggerConfig()
pos = PosTagger(config=config)
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
from bgnlp import LemmaTaggerConfig, LemmaTagger

lemma = LemmaTagger(config=LemmaTaggerConfig())
text = "Добре дошли!"
print(lemma(text))
```

```bash
[{'word': 'Добре', 'lemma': 'Добре'}, {'word': 'дошли', 'lemma': 'дойда'}, {'word': '!', 'lemma': '!'}]
```
