# TODO: Change .pos to .lemma
from bgnlp.tools.pos import BgLemmatizer
from bgnlp.tools.configs import BgLemmatizerConfig
from bgnlp.tools.vocabs import get_vocab


vocab = get_vocab("symbols-vocab")
config = BgLemmatizerConfig(device="cpu")

input_words = "Сестрата му беше на високо в планината"
lemmatizer = BgLemmatizer(config=config, vocab=vocab)
prediction = lemmatizer(
    words=input_words
)
print(f"Input: {input_words}")
print(f"Prediction: {prediction}")
