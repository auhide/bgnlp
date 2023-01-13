from bg_nlp.tools.pos import BgLemmatizer
from bg_nlp.tools.configs import BgLemmatizerConfig
from bg_nlp.tools.vocabs import get_vocab


vocab = get_vocab("symbols-vocab")
config = BgLemmatizerConfig(device="cpu")

input_words = "Животът не ти дава мира знаеш ли"
lemmatizer = BgLemmatizer(config=config, vocab=vocab)
prediction = lemmatizer(
    words=input_words
)
print(f"Input: {input_words}")
print(f"Prediction: {prediction}")