import os

import pandas as pd
import torch

from bg_nlp.tools.vocabs import get_vocab
from bg_nlp.tools.configs import BgLemmatizerConfig
from bg_nlp.tools.tokenizers import DefaultTokenizer
from bg_nlp.lib.datasets import LemmatizationDataset
from bg_nlp.tools.trainers import BgLemmatizerTrainer


# The path to the dataset. You can download it from this link and test it out
# yourself!
DATASET_PATH = os.path.join("..", "datasets", "bg-pos", "bg-pos.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initializing the model training configuration.
config = BgLemmatizerConfig(device=DEVICE)

# CSV dataset with these two columns:
# 'word' - the words
# 'lemma' - the lemmas of the respective words
dataset_df = pd.read_csv(DATASET_PATH, sep="\t")[:10]

# Initializing the pre-created Vocabulary and Tokenizer.
vocab = get_vocab("symbols-vocab")
tokenizer = DefaultTokenizer()

# Initializing the Dataset.
dataset = LemmatizationDataset(
    words=dataset_df["word"].tolist(), 
    lemmas=dataset_df["lemma"].tolist(),
    tokenizer=tokenizer, 
    vocab=vocab,
    words_max_size=config.max_word_size, 
    lemmas_max_size=config.max_lemma_size
)

lemma_trainer = BgLemmatizerTrainer(
    dataset=dataset, 
    tokenizer=tokenizer, 
    vocab=vocab
)
lemma_trainer.train(
    config=config, 
    log_words=[
        "конете", 
        "комарът", 
        "кравите", 
        "патриот",
    ]
)
