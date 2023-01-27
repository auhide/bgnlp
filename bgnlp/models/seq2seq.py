import logging
import random

import torch
from torch import nn
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader, Dataset


logging.basicConfig(
    format="%(asctime)s|%(levelname)s| %(message)s", 
    level=logging.INFO
)


class Seq2Seq(nn.Module):
    
    def __init__(
        self, 
        encoder: torch.nn.Module, 
        decoder: torch.nn.Module, 
        reverse_input=True
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

        self.reverse_input = reverse_input

    def forward(self, x: torch.Tensor, y: torch.Tensor, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = y.shape[0], y.shape[1]
        vocab_size = self.decoder.vocab_size

        # Tensor in which we are going to store the Decoder outputs.
        outputs = torch.zeros(size=(batch_size, tgt_len, vocab_size))

        # Reversing the sequences in the input tensor.
        # In the paper it's stated that it is extremely valuable, and makes a difference.
        if self.reverse_input:
            x = torch.flip(x, [1])
        
        enc_out, hidden = self.encoder(x)

        # The start token:
        curr_input = y[:, 0]

        for t in range(1, tgt_len):
            out, hidden = self.decoder(curr_input, hidden, enc_out)
            outputs[:, t, :] = out

            teacher_force = teacher_forcing_ratio > random.random()

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
        padding_idx: int, 
        bidirectional=True, 
        dropout=0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = 1
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            input_size=embed_size, 
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)

        directions = 2 if self.bidirectional else 1
        
        # A fully-connected layer used for the generation of a single hidden state.
        # It's used since we have twice as many hidden states (we are using a BiLSTM).
        self.fc = nn.Linear(directions * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor):
        embedded_x = self.dropout(self.embed(x))
        out, hidden = self.gru(embedded_x)
        # out.shape: (BATCH_SIZE, SEQ_LEN, HID_SIZE)

        # Concatenating the hidden states in both directions along the 2nd dimension.
        hidden = torch.tanh(
            self.fc(
                torch.cat(
                    (hidden[-2, :, :], hidden[-1, :, :]), 
                    dim=1
                )
            )
        )
        # hidden.shape: (BATCH_SIZE, HIDDEN_DIM)

        return out, hidden


class Attention(nn.Module):
    
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int):
        super().__init__()
        
        self.attn = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden.shape: (BATCH_SIZE, DEC_HID_DIM)
        # encoder_outputs.shape: (BATCH_SIZE, SEQ_LEN, ENC_HID_DIM * 2)
        seq_len = encoder_outputs.shape[1]

        # Repeating the previous decoder hidden state 'seq_len' times, since
        # the decoder's hidden state doesn't have a SEQ_LEN dimension.
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # After the change hidden.shape will be (BATCH_SIZE, SEQ_LEN, DEC_HID_DIM)

        # Calculating the energy between the prev. decoder hidden state and 
        # the hidden state of the encoder.
        energy = torch.sigmoid(
            self.attn(
                torch.concat((hidden, encoder_outputs), dim=-1)
            )
        )

        # Projecting the energy of all sequences in the batch to a single sequence
        # of shape (SEQ_LEN, 1).
        a = self.v(energy).squeeze(-1)
        # Hence, our attention tensor 'a' should be of shape (BATCH_SIZE, SEQ_LEN).

        # Softmaxing along the SEQ_LEN dimension.
        return torch.softmax(a, dim=1)


class Decoder(nn.Module):

    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int, 
        hidden_size: int, 
        padding_idx: int, 
        attention: nn.Module,
        dropout=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.num_layers = 1

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            input_size=(2 * hidden_size) + embed_size, 
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        self.project_out = nn.Linear(2 * hidden_size + hidden_size + embed_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, enc_out: torch.Tensor):
        # x.shape: (BATCH_SIZE, SEQ_LEN)
        # hidden.shape: (BATCH_SIZE, EMB_SIZE)
        # enc_out.shape: (BATCH_SIZE, DEC_EMB_SIZE * 2)

        x = x.unsqueeze(1)

        embedded_x = self.dropout(self.embed(x))
        # embedded_x.shape: (BATCH_SIZE, SEQ_LEN, EMB_SIZE)
        a = self.attention(hidden, enc_out)
        # Adding a 2nd dimension, since torch.bmm expects the two tensors to be
        # 3D.
        a = a.unsqueeze(1)
        # a.shape: (BATCH_SIZE, 1, SEQ_LEN)
        
        # The weighted sum of the encoder hidden states, using the attention
        # tensor.
        weighted = a @ enc_out
        # weighted.shape: (BATCH_SIZE, 1, (2 * ENC_HID_SIZE))

        gru_input = torch.cat((embedded_x, weighted), dim=-1)
        # gru_input.shape: (BATCH_SIZE, 1, (2 * ENC_HID_SIZE) + DEC_EMB_SIZE)

        # Since gru_input is batched (is 3D), hidden should also be batched, 
        # hence, we add an empty SEQ_LEN dimension.
        out, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        # out.shape: (BATCH_SIZE, 1, DEC_HID_SIZE)
        # hidden.shape: (1, BATCH_SIZE, DEC_HID_SIZE)
        
        # Removing the SEQ_LEN dimension
        embedded_x = embedded_x.squeeze(1)
        out = out.squeeze(1)
        weighted = weighted.squeeze(1)

        pred = self.project_out(torch.cat((out, weighted, embedded_x), dim=-1))
        # pred.shape: (BATCH_SIZE, VOCAB_SIZE)

        return pred, hidden.squeeze(0)


class Seq2SeqSession:

    def __init__(
        self, 
        model: torch.nn.Module, 
        loss: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        vocab: Vocab, 
        device="cpu", 
        clip=5
    ):
        self.model = model
        self.loss_func = loss
        self.optimizer = optimizer

        self.vocab = vocab
        self.device = device
        self.clip = clip

        # Index to token/string.
        self.itos = {self.vocab[token]: token for token in self.vocab.vocab.itos_}
        # String/token to index.
        self.stoi = vocab

        self._init_weights(self.model)

    def train(
        self, 
        train_dataset: Dataset, valid_dataset: Dataset, 
        batch_size: int, epochs: int = 1, fixed_input: torch.LongTensor = None,
        num_workers: int = 0, teacher_forcing_ratio: float = 0.5, metrics: dict = None
    ):
        self.teacher_forcing_ratio = teacher_forcing_ratio

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
        self.metrics = metrics

        for epoch in range(epochs):
            train_loss, train_metrics = self._train_epoch(train_dataloader)
            valid_loss, valid_metrics = self._valid_epoch(valid_dataloader)

            logging.info(f"Epoch: {epoch + 1}/{epochs}, Training Loss: {train_loss:.2f}, Validation Loss: {valid_loss:.2f}")
            self._log_metrics("Training", metrics=train_metrics)
            self._log_metrics("Validation", metrics=valid_metrics)
            logging.info("")

        return self.model

    def predict(self, word: torch.LongTensor):
        if len(word.shape) == 1:
            word = word.unsqueeze(0)

        seq_len = word.shape[-1]

        # Target with only a [START] token (and padding).
        target = [self.vocab["[START]"]] + [self.vocab["[PAD]"]] * (seq_len - 1)
        # Repeat the 'target' tensor along the row axis 'batch_size' times.
        target = torch.LongTensor(target).repeat(word.shape[0], 1).to(self.device)

        # Setting teacher_forcing_ratio to 0, since we want to use the prediction
        # generated by the model as the next input to the decoder.
        lemma = self.model(word, target, teacher_forcing_ratio=0)
        
        return lemma.argmax(-1)

    def _train_epoch(self, dataloader: DataLoader):
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            y_pred = self.model(
                x, y, 
                teacher_forcing_ratio=self.teacher_forcing_ratio
            ).to(self.device)
            # y_pred's shape - (batch_size, seq_len, vocab_size)
            
            # Reshaping y_pred:
            vocab_size = y_pred.shape[-1]
            y_pred = y_pred[:, 1:, :].reshape(-1, vocab_size)

            # Reshaping y:
            y = y[:, 1:].reshape(-1)
            
            self.optimizer.zero_grad()

            loss = self.loss_func(y_pred, y)
            loss.backward()

            # Clipping the gradients, since LSTMs/GRUs have Exploding Gradient problems.
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

        metrics = None
        if self.metrics is not None:
            metrics = self._get_metrics_results(y_pred, y)

        return loss.item(), metrics

    def _valid_epoch(self, dataloader: DataLoader):
        self.model.eval()

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                y_pred = self.model(x, y, teacher_forcing_ratio=0).to(self.device)

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

                logging.info(f"Inputs:\n{fixed_input}")
                logging.info(f"Predictions:\n{fixed_pred}")

        metrics = None
        if self.metrics is not None:
            metrics = self._get_metrics_results(y_pred, y)

        self.model.train()

        return loss.item(), metrics

    def _init_weights(self, m: nn.Module):
        # This initialization comes from the Seq2Seq paper.
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def _get_metrics_results(self, y_pred: torch.Tensor, y: torch.Tensor):
        results = {}

        for metric_name, metric in self.metrics.items():
            y_pred_argmax = y_pred.argmax(-1)
            result = metric(y_pred_argmax, y)

            results[metric_name] = float(result)

        return results

    def _log_metrics(self, type_: str, metrics: dict):
        log_messages = []
        for metric_name, metric in metrics.items():
            log_messages.append(f"{type_} {metric_name}: {metric:.2f}")

        logging.info(", ".join(log_messages))
