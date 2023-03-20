from bgnlp import PosTagger, PosTaggerConfig


config = PosTaggerConfig()
pos = PosTagger(config=config)
text = "Това е библиотека за обработка на естествен език."

print(f"Input: {text}")
print("Result:", pos(text))
