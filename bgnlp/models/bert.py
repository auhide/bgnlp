import torch
from torch import nn
from transformers import BertModel, BertConfig


class LemmaBert(nn.Module):
    
    def __init__(
        self, 
        vocab_size: int, 
        output_size: int, 
        hidden_size: int = 512, 
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        intermediate_size=1028,
        device="cpu"
    ):
        super().__init__()
        
        self.base_model = BertModel(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size
        )).to(device)
        self.base_model.resize_token_embeddings(output_size)
        self.device = self.base_model.device
        
        self.project = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_ids: torch.LongTensor, attention_mask=None) -> torch.Tensor:
        out = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        return self.project(out)
