from bgnlp import lemmatize
from bgnlp import LemmaTaggerConfig, LemmaTagger


text = "Добре дошли!"
print("Input:", text)

# You can either do this:
print("Output:", lemmatize(text))

# Or this:
lemma = LemmaTagger(config=LemmaTaggerConfig())
print("Output:", lemma("Добре дошли"))
