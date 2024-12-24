from typing import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch

from doe.models.blocks import FeedForwardLayer, DecoderBlock


class ChickenAdvanced(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        step_size: int = 8,  # Max num characters to look back.
        embedding_dim: int = 32,
        n_blocks: int = 3,
        n_heads: int = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.step_size: int = step_size
        # This is a lookup table. Token embedding table. (V x dim)
        self.token_embedding_table: torch.nn.modules.sparse.Embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
        )
        # Also positional embedding table. (B x step x dim)
        self.positional_embedding_table: torch.nn.modules.sparse.Embedding = (
            nn.Embedding(
                num_embeddings=step_size,
                embedding_dim=embedding_dim,
            )
        )
        # Why is this linear layer needed? It converts token embeddings to
        self.lm_head = nn.Linear(
            in_features=embedding_dim, out_features=vocabulary_size
        )

        self.decoder_blocks = nn.Sequential(
            *[
                DecoderBlock(
                    step_size=step_size,
                    embedding_dim=embedding_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.feedforward = FeedForwardLayer(
            embedding_dim=embedding_dim, dropout=dropout
        )

    def forward(self, index, target=None):
        batch_size, step_size = index.shape
        # print(index.shape, step_size, torch.arange(step_size))

        # Use the index to retrieve a tensor.
        token_ebd = self.token_embedding_table(
            index
        )  # With shape (batch_size, step/time, channel/dim).
        pos_ebd = self.positional_embedding_table(
            torch.arange(step_size)
        )  # (step, dim)

        # Embeddings combined.
        combined = token_ebd + pos_ebd

        # These contains a series of decoder blocks. Each block has
        # Feed to self-attention layer.
        # and a feedforward layer
        # combined = self.self_att_head(combined)
        # combined = self.feedforward(combined)
        combined = self.decoder_blocks(combined)
        combined = self.layer_norm(combined)

        logits = self.lm_head(
            combined
        )  # From (batch, step, dim) to (batch, step, vocabulary_size)

        if target is not None:
            batch_size, step, dim = logits.shape
            # Pytorch expects B, D, S as logits.
            # And B, D as targets.
            # Define the loss function.
            loss = F.cross_entropy(
                logits.view(batch_size * step, dim),
                target.view(batch_size * step),
            )
        else:
            loss = None
        return logits, loss

    # Generates output.
    def generate(self, index, output_length: int) -> torch.Tensor:
        # Index will be (batch_size, step).
        for _ in range(output_length):
            index_cropped = index[:, -self.step_size :]
            logits, _ = self(index_cropped)
            # Here logits will return as (batch_size, step, dim). We want to keep only the last step.
            probs = F.softmax(
                logits[
                    :, -1, :
                ],  # Keeps only the last step, so now the shape is (batch, dim).
                dim=-1,  # ???
            )

            # Sample from the distribution ???????
            predicted_index = torch.multinomial(
                probs, num_samples=1
            )  # Shape will be (batch, 1).
            # Now append the result.
            index = torch.concat((index, predicted_index), dim=1)
        return index
