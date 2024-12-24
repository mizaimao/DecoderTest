from typing import Dict
import torch.nn as nn
from torch.nn import functional as F
import torch


class Head(nn.Module):
    """Single-head self-attention block with mask."""
    def __init__(
        self,
        head_size: int,
        embedding_dim: int,
        step_size: int,
        dropout: float,
    ):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones((step_size, step_size))))
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        batch_size, step_size, dim = x.shape
        # Note heere k and q are coming from the same source.
        k = self.key(x)  # (batch, step, dim).
        q = self.query(x)  # (batch, step, dim).
        weights = q @ k.transpose(-2, -1) * (dim ** -0.5)  # (batch, step, head) @ (batch, head, step) -> (batch, step, step)
        weights = weights.masked_fill(self.tril[:step_size, :step_size] == 0, float('-inf'))  # (batch, step, step)
        weights = F.softmax(weights, dim=-1)  # (batch, step, step)

        weights = self.dropout(weights)
        
        v = self.value(x)  # (batch, step, dim)
        out = weights @ v  # (batch, step, step) @ (batch, step, dim) -> (batch, step, dim)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention wrapper."""
    def __init__(
        self,
        n_heads: int,
        head_size: int,
        embedding_dim: int,
        step_size: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(head_size=head_size, step_size=step_size, embedding_dim=embedding_dim,dropout=dropout)
                for _ in range(n_heads)
            ]
        )
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim = -1)
        out = self.projection(out)  # Just a simple linear projection of the outcome.
        out = self.dropout(out)
        return out


class FeedForwardLayer(nn.Module):
    """A place for nodes to exchange what they learned..."""
    def __init__(self, embedding_dim: int, dropout: float = 0.5, magic_multiplier: int = 4):
        """
        That magic number is inhereted from the original Transformer... alchemy.
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * magic_multiplier),
            nn.ReLU(),
            nn.Linear(embedding_dim * magic_multiplier, embedding_dim),  # <- this is a projection layer.
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layer(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        step_size: int = 8,  # Max num characters to look back.
        embedding_dim: int = 32,
        n_heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Single head block then.
        if n_heads is None:
            self.self_att_head = Head(head_size=head_size, step_size=step_size, embedding_dim=embedding_dim)
        else:
            split_head_size: int = embedding_dim // n_heads
            assert split_head_size == embedding_dim / n_heads
    
            self.self_att_head = MultiHeadAttention(
                step_size=step_size, head_size=split_head_size, embedding_dim=embedding_dim, n_heads=n_heads, dropout=dropout
            )

        self.feedforward = FeedForwardLayer(embedding_dim=embedding_dim)
        self.norm_layer_1 = nn.LayerNorm(embedding_dim)
        self.norm_layer_2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Note originally layer normalization are applied AFTER the attention and feedforward. 
        # But it's now a more popular ??!???!?!? idea to apply them BEFORE.
        x = self.norm_layer_1(x)
        x = self.self_att_head(x) + x  # The addition is residual.
        x = self.norm_layer_2(x)
        x = self.feedforward(x) + x  # Same here, residual is added.
        return x

@torch.no_grad()
def get_valid_loss(
    model: nn.Module,
    loader
):
    model.eval()
    x, y = loader.get_data("valid")
    _logits, loss = model(x, y)

    model.train()
    return loss
