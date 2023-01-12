import os

import torch
import pandas as pd

from bg_nlp.tokenizers import DefaultTokenizer
from bg_nlp.datasets import LemmatizationDataset
from bg_nlp.models.seq2seq import Attention, Encoder, Decoder, Seq2Seq, Seq2SeqSession
from bg_nlp.preprocessing import train_validation_split


bg_tokenizer = DefaultTokenizer()

vocab = torch.load(os.path.join(".", "bg_nlp", "vocabs", "svocab.pt"))

DATASET_PATH = os.path.join("..", "datasets", "bg-pos", "bg-pos.csv")

df = pd.read_csv(DATASET_PATH, sep="\t")

# You may plot the histograms for these columns to see the distributions
# of the words and their lemmas.
df["word_size"] = df["word"].apply(lambda x: len(bg_tokenizer(x, split_type="symbol")))
df["lemma_size"] = df["lemma"].apply(lambda x: len(bg_tokenizer(x, split_type="symbol")))

# Biggest word is of size 13 symbols.
# I am going to choose maximum sizes ot 15, since we count the [START] and [END]
# tokens as well.
dataset = LemmatizationDataset(
    df["word"].tolist(), df["lemma"].tolist(),
    tokenizer=bg_tokenizer, vocab=vocab,
    words_max_size=15, lemmas_max_size=15
)

train_dataset, valid_dataset = train_validation_split(dataset, train_size=0.9)

enc = Encoder(
    vocab_size=len(vocab),
    embed_size=256,
    hidden_size=256,
    padding_idx=vocab["[PAD]"],
    dropout=0.5
)

attn = Attention(encoder_hidden_dim=256, decoder_hidden_dim=256)

dec = Decoder(
    vocab_size=len(vocab),
    embed_size=256,
    hidden_size=256,
    padding_idx=vocab["[PAD]"],
    attention=attn,
    dropout=0.5
)


L_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIXED_WORDS = [
    "конете", "комарът", "кравите", "патриот"
]
FIXED_DATASET = LemmatizationDataset(
    FIXED_WORDS, None,
    tokenizer=bg_tokenizer, vocab=vocab,
    words_max_size=15, lemmas_max_size=15
)

MODEL_PATH = os.path.join(".", "bg_nlp", "weights", "seq2seq.pt")


seq2seq = Seq2Seq(encoder=enc, decoder=dec).to(DEVICE)

optimizer = torch.optim.Adam(params=seq2seq.parameters(), lr=L_RATE)
loss = torch.nn.CrossEntropyLoss().to(DEVICE)

print("Starting training session...")
print(f"Device: {DEVICE}")

trainer = Seq2SeqSession(
    model=seq2seq, 
    loss=loss, 
    optimizer=optimizer,
    vocab=vocab,
    device=DEVICE
)
seq2seq = trainer.train(
    train_dataset=dataset, valid_dataset=valid_dataset,
    batch_size=BATCH_SIZE, epochs=EPOCHS,
    fixed_input=FIXED_DATASET.x.to(DEVICE)
)


# Saving the trained model.
print(f"Saving model to {MODEL_PATH}")
torch.save(seq2seq.state_dict(), MODEL_PATH)
