"""
Transformer building blocks
"""
from typing import Optional

import torch
import torch.nn as nn

from attention import MultiAttentionHead


class FFN(nn.Module):
    def __init__(self, input_size: int, inner_size: int) -> None:
        super().__init__()
        self.f1 = nn.Linear(input_size, inner_size)
        self.f2 = nn.Linear(inner_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f2(torch.relu(self.f1(x)))


class Residual(nn.Module):
    def __init__(self, inner_module: nn.Module) -> None:
        """
        Note: expect the inner module to return a single output which has
        the same size as the input tensor
        """
        super().__init__()
        self.module = inner_module

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.module(x, **kwargs) + x


class EncoderBlock(nn.Module):
    """
    Self-attention and feed-forward module
    """

    def __init__(
        self,
        input_size: int,
        ff_size: int,
        key_size: int,
        value_size: int,
        n_heads: int,
    ):
        super().__init__()
        self.attention_heads = Residual(
            MultiAttentionHead(
                input_size, key_size, value_size, input_size, n_heads
            )
        )
        self.ffn = Residual(FFN(input_size, ff_size))

        # Normalise over last dimension only
        self.ln1 = nn.LayerNorm((input_size))
        self.ln2 = nn.LayerNorm((input_size))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.ln1(self.attention_heads(x))
        x = self.ln2(self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        ff_size: int,
        key_size: int,
        value_size: int,
        n_attention_heads: int,
        n_layers: int,
    ):
        super().__init__()
        self.encoder_blocks = nn.Sequential(
            *[
                EncoderBlock(
                    input_size,
                    ff_size,
                    key_size,
                    value_size,
                    n_attention_heads,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_layers(x)
