from bgnlp.tools.taggers import LemmaTagger
from bgnlp.tools.configs import LemmaTaggerConfig


if __name__ == "__main__":
    lemma = LemmaTagger(config=LemmaTaggerConfig)
    text = "Добре дошли!"
    print("Input:", text)
    print("Output:", lemma("Добре дошли"))
