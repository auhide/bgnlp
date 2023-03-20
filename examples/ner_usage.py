from bgnlp import NerTagger, NerTaggerConfig


ner = NerTagger(config=NerTaggerConfig())
text = "Барух Спиноза е роден в Амстердам"

print(f"Input: {text}")
print("Result:", ner(text))
