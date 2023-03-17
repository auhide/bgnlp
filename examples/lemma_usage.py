from bgnlp import LemmaTaggerConfig, LemmaTagger


if __name__ == "__main__":
    lemma = LemmaTagger(config=LemmaTaggerConfig())
    text = "Добре дошли!"
    print("Input:", text)
    print("Output:", lemma("Добре дошли"))
