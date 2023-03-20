from bgnlp import LemmaTaggerConfig, LemmaTagger


lemma = LemmaTagger(config=LemmaTaggerConfig())
text = "Добре дошли!"
print("Input:", text)
print("Output:", lemma("Добре дошли"))
