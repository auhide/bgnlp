from bgnlp import PosTagger, PosTaggerConfig


config = PosTaggerConfig()
pos = PosTagger(config=config)
print(pos("Това е библиотека за обработка на естествен език."))
