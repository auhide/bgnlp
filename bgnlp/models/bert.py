from torch import nn
from transformers import RobertaModel, RobertaConfig


# TODO: Add the model trainer.
class BgPosRoBerta(nn.Module):
    
    def __init__(self, vocab, n_classes, dropout=0.1):
        super().__init__()
        
        hidden_size = 126

        self.base_model = RobertaModel(
            RobertaConfig(
                vocab_size=len(vocab),
                hidden_size=hidden_size,
                num_hidden_layers=6,
                num_attention_heads=6,
                intermediate_size=512,
                pad_token_id=vocab["[PAD]"]
            )
        )
        
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(hidden_size, n_classes)
        
    def forward(self, x):
        # Taking the tensor representing the [CLS] token of the
        # output. Since it contains all the information about the previous tokens
        # (the ones from the input), it is sufficient for classification objectives.
        out = self.base_model(x).pooler_output
        # Projecting the output of RoBERTa onto the dimension of classes.
        out = self.project(self.dropout(out))
        
        return out
