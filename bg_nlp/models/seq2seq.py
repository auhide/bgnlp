import re
import random

import torch
from torch import nn
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader, Dataset

from bg_nlp.tokenizers import Tokenizer


class Seq2Seq(nn.Module):
    
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, reverse_input=True):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

        self.reverse_input = reverse_input

    def forward(self, x: torch.Tensor, y: torch.Tensor, teacher_forcing_ration=0.5):
        batch_size, tgt_len = y.shape[0], y.shape[1]
        vocab_size = self.decoder.vocab_size

        # Tensor in which we are going to store the Decoder outputs.
        outputs = torch.zeros(size=(batch_size, tgt_len, vocab_size))

        # Reversing the sequences in the input tensor.
        # In the paper it's stated that it is extremely valuable, and makes a difference.
        if self.reverse_input:
            x = torch.flip(x, [1])
        
        _, hidden = self.encoder(x)

        # The start token:
        curr_input = y[:, 0]

        for t in range(1, tgt_len):
            out, hidden = self.decoder(curr_input, hidden)
            outputs[:, t, :] = out

            teacher_force = teacher_forcing_ration > random.random()

            # Getting the top prediction.
            top_pred = out.argmax(1)

            # If we are in a teacher forcing state - use actual next token
            # as an input to the Decoder, else - use the top prediction.
            curr_input = y[:, t] if teacher_force else top_pred

        return outputs


class Encoder(nn.Module):

    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int, 
        hidden_size: int, 
        num_layers: int, 
        padding_idx: int,
        bidirectional=True,
        dropout=0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        embedded_x = self.dropout(self.embed(x))
        out, hidden = self.lstm(embedded_x)

        return out, hidden


class Decoder(nn.Module):

    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int, 
        hidden_size: int, 
        num_layers: int, 
        padding_idx: int,
        bidirectional=True,
        dropout=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)

        directions = 2 if bidirectional else 1
        self.project = nn.Linear(directions * hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: tuple):
        x = x.unsqueeze(1)

        embedded_x = self.dropout(self.embed(x))
        hidden = [state.detach() for state in hidden]
        out, hidden = self.lstm(embedded_x, hidden)
        out = self.project(out)

        return out.squeeze(1), hidden


class Attention(nn.Module):
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int, 
        hidden_size: int, 
        padding_idx: int, 
        max_len, 
        dropout=0.1
    ):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        
        self.attn = nn.Linear(embed_size * 2, max_len)
        
    def forward(self, x, hidden, encoder_outputs):
        pass


class Seq2SeqSession:

    def __init__(self, model, loss, optimizer, vocab, device="cpu", clip=5):
        self.model = model
        self.loss_func = loss
        self.optimizer = optimizer

        self.vocab = vocab
        self.device = device

        # Index to token/string.
        self.itos = {self.vocab[token]: token for token in self.vocab.vocab.itos_}
        # String/token to index.
        self.stoi = vocab

        self.clip = clip

        self._init_weights(self.model)

    def train(
        self, 
        train_dataset: Dataset, valid_dataset: Dataset, 
        batch_size: int, epochs: int = 1, fixed_input: torch.LongTensor = None,
        num_workers: int = 0
    ):
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True
        )
        self.fixed_input = fixed_input

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_dataloader)
            valid_loss = self._valid_epoch(valid_dataloader)

            print(f"Epoch: {epoch + 1}, Training Loss: {train_loss:.2f}, Validation Loss: {valid_loss:.2f}")

    def predict(self, word: torch.LongTensor):
        if len(word.shape) == 1:
            word = word.unsqueeze(0)

        seq_len = word.shape[-1]

        # Target with only a [START] token (and padding).
        target = [self.vocab["[START]"]] + [self.vocab["[PAD]"]] * (seq_len - 1)
        # Repeat the 'target' tensor along the row axis 'batch_size' times.
        target = torch.LongTensor(target).repeat(word.shape[0], 1).to(self.device)

        # Setting teacher_forcing_ration to 0, since we want to use the prediction
        # generated by the model as the next input to the decoder.
        lemma = self.model(word, target, teacher_forcing_ration=0)
        
        return lemma.argmax(-1)

    def _train_epoch(self, dataloader: DataLoader):
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            y_pred = self.model(x, y).to(self.device)
            # y_pred's shape - (batch_size, seq_len, vocab_size)
            
            # Reshaping y_pred:
            vocab_size = y_pred.shape[-1]
            y_pred = y_pred[:, 1:, :].reshape(-1, vocab_size)

            # Reshaping y:
            y = y[:, 1:].reshape(-1)
            
            self.optimizer.zero_grad()

            loss = self.loss_func(y_pred, y)
            loss.backward()

            # Clipping the gradients, since LSTMs have Exploding Gradient problems.
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

        return loss.item()

    def _valid_epoch(self, dataloader):
        self.model.eval()

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                y_pred = self.model(x, y).to(self.device)

                vocab_size = y_pred.shape[-1]
                y_pred = y_pred[:, 1:, :].reshape(-1, vocab_size)

                y = y[:, 1:].reshape(-1)

                loss = self.loss_func(y_pred, y)

            if self.fixed_input is not None:
                fixed_pred = self.predict(word=self.fixed_input)

                fixed_input = self.fixed_input.tolist()
                fixed_inputs = []
                for seq in fixed_input:
                    fixed_inputs.append("".join([self.itos[int(idx)] for idx in seq]))

                fixed_preds = []
                for seq in fixed_pred:
                    fixed_preds.append("".join([self.itos[int(idx)] for idx in seq]))

                fixed_input = '\n'.join(fixed_inputs)
                fixed_pred = '\n'.join(fixed_preds)

                print(f"Input:\n{fixed_input}")
                print(f"Prediction:\n{fixed_pred}")
                print()

        self.model.train()

        return loss.item()

    def _init_weights(self, m):
        # This initialization comes from the Seq2Seq paper.
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
