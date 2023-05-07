from bgnlp import pos
from bgnlp import PosTagger, PosTaggerConfig


text = "Това е библиотека за обработка на естествен език."
print(f"Input: {text}")

# You can either do this:
print("Result:", pos(text))

# Or this:
config = PosTaggerConfig()
pos = PosTagger(config=config)
print("Result:", pos(text))
