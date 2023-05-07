from bgnlp import ner
from bgnlp import NerTagger, NerTaggerConfig


text = "Барух Спиноза е роден в Амстердам"
print(f"Input: {text}")

# You can either do this:
print("Result", ner(text))

# Or this:
ner = NerTagger(config=NerTaggerConfig())
print("Result:", ner(text))
