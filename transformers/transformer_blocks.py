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
        self.attention_heads = MultiAttentionHead(
            input_size, key_size, value_size, input_size, n_heads
        )
        self.ffn = FFN(input_size, ff_size)

        # Normalise over last dimension only
        self.ln1 = nn.LayerNorm((input_size))
        self.ln2 = nn.LayerNorm((input_size))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # input gets multiplied three times to q, k, v
        x = self.ln1(self.attention_heads(x, x, x) + x)
        x = self.ln2(self.ffn(x) + x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        ff_size: int,
        key_size: int,
        value_size: int,
        n_heads: int,
        n_layers: int,
    ):
        super().__init__()
        self.encoder_blocks = nn.Sequential(
            *[
                EncoderBlock(
                    input_size, ff_size, key_size, value_size, n_heads,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_layers(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        ff_size: int,
        key_size: int,
        value_size: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.mah1 = MultiAttentionHead(
            input_size, key_size, value_size, input_size, n_heads
        )

        self.mah2 = MultiAttentionHead(
            input_size, key_size, value_size, input_size, n_heads
        )

        self.ffn = FFN(input_size, ff_size)
        self.ln1 = nn.LayerNorm((input_size))
        self.ln2 = nn.LayerNorm((input_size))
        self.ln3 = nn.LayerNorm((input_size))

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.ln1(self.mah1(x, x, x, mask) + x)
        out = self.ln2(self.mah2(out, encoder_output, encoder_output) + out)
        out = self.ln3(self.ffn(out) + out)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        ff_size: int,
        key_size: int,
        value_size: int,
        n_heads: int,
        n_layers: int,
    ) -> None:

        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    input_size, ff_size, key_size, value_size, n_heads,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for decoder in self.decoder_blocks:
            x = decoder(x, encoder_output, mask)
        return x
