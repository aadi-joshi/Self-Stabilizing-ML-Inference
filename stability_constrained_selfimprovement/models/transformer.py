# ============================================================================
# Small Transformer for Algorithmic Tasks (Experiment B)
# Supports copy, reverse, and sort sequence-to-sequence tasks
# ============================================================================

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AlgorithmicTransformer(nn.Module):
    """
    Small encoder-decoder transformer for algorithmic sequence tasks.
    
    Tasks:
        - copy: output = input
        - reverse: output = reversed input
        - sort: output = sorted input
    
    Architecture:
        Embedding → PositionalEncoding → TransformerEncoder → Linear → Vocab logits
    """

    def __init__(
        self,
        vocab_size: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_seq_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.output_proj.bias.data.zero_()
        self.output_proj.weight.data.uniform_(-init_range, init_range)

    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """Return intermediate representations for drift measurement."""
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        hidden = self.transformer_encoder(emb)
        return hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) integer tokens
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        hidden = self.get_representations(x)
        logits = self.output_proj(hidden)
        return logits


def generate_algorithmic_data(
    task: str,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple:
    """
    Generate (input, target) pairs for algorithmic tasks.
    
    Args:
        task: One of 'copy', 'reverse', 'sort'
        batch_size: Number of sequences
        seq_len: Length of each sequence
        vocab_size: Number of distinct tokens (values 0..vocab_size-1)
        device: Torch device
        
    Returns:
        (input_seq, target_seq) each of shape (batch_size, seq_len)
    """
    # Generate random integer sequences
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    if task == "copy":
        y = x.clone()
    elif task == "reverse":
        y = x.flip(dims=[1])
    elif task == "sort":
        y, _ = x.sort(dim=1)
    else:
        raise ValueError(f"Unknown task: {task}. Choose from ['copy', 'reverse', 'sort']")

    return x, y
