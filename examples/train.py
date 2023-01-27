import os
from dataclasses import dataclass

import torch
import pandas as pd
from torchmetrics.classification import Accuracy, F1Score

from bgnlp.tools.vocabs import get_vocab
from bgnlp.tools.configs import BgLemmatizerConfig
from bgnlp.tools.tokenizers import DefaultPreTokenizer
from bgnlp.lib.datasets import LemmatizationDataset
from bgnlp.tools.trainers import BgLemmatizerTrainer


# Creating a custom config and overloading some values.
@dataclass
class CustomLemmatizerConfig(BgLemmatizerConfig):
    epochs: int = 10


# Setting a constant pseudo-randomness state.
SEED = 42
torch.backends.cudnn.benchmark = True
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


# The path to the dataset. You can download it from this link and test it out
# yourself!
DATASET_PATH = os.path.join("..", "datasets", "bg-pos", "bg-pos.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initializing the model training configuration.
config = CustomLemmatizerConfig(device=DEVICE)

# CSV dataset with these two columns:
# 'word' - the words
# 'lemma' - the lemmas of the respective words
dataset_df = pd.read_csv(DATASET_PATH, sep="\t")

# Initializing the pre-created Vocabulary and Tokenizer.
vocab = get_vocab("symbols-vocab")
tokenizer = DefaultPreTokenizer()

# Initializing the Dataset.
dataset = LemmatizationDataset(
    words=dataset_df["word"].tolist(), 
    lemmas=dataset_df["lemma"].tolist(),
    tokenizer=tokenizer, 
    vocab=vocab,
    words_max_size=config.max_word_size, 
    lemmas_max_size=config.max_lemma_size
)

# Defining and starting the Trainer.
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
        "Търново", 
        "катерачи",
        "беше",
        "в",
        "във",
        "с",
        "със",
        "съм",
        "кон",
    ],
    metrics={
        "Accuracy": Accuracy(
            task="multiclass", 
            num_classes=len(vocab),
            ignore_index=vocab["[PAD]"]
        ).to(DEVICE),
        "F1-Score": F1Score(
            task="multiclass", 
            num_classes=len(vocab),
            ignore_index=vocab["[PAD]"]
        ).to(DEVICE),
    }
)
