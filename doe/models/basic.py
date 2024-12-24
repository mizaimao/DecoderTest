"""
This is a really simple model having only one table. And of course the perforamnce is pretty bad.
Outputs are basically rubbish.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class ChickenSimple(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        # This is a lookup table.
        self.token_embedding_table: torch.nn.modules.sparse.Embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=vocabulary_size,
        )

    def forward(self, index, target = None):
        # Use the index to retrieve a tensor.
        logits = self.token_embedding_table(index)  # With shape (batch_size, step/time, channel/dim).

        
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
            logits, _ = self(index)
            # Here logits will return as (batch_size, step, dim). We want to keep only the last step.
            probs = F.softmax(
                logits[:, -1, :],  # Keeps only the last step, so now the shape is (batch, dim).
                dim=-1,  # ???
            )

            # Sample from the distribution ???????
            predicted_index = torch.multinomial(probs, num_samples=1)  # Shape will be (batch, 1).
            # Now append the result.
            index = torch.concat((index, predicted_index), dim=1)
        return index
        

m = ChickenSimple(vocabulary_size)
out = m(x_batch, y_batch)
# print(out[0].shape, out[1])
# print(out)
        
        