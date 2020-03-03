import torch
from torch import tanh
import torch.nn as nn

from rnns.helpers import RNNGate


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer = RNNGate(input_size + hidden_size, hidden_size, tanh)

    def _init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), dtype=torch.float)

    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size = x.size(0)
            hidden = self._init_hidden(batch_size)

        x = torch.cat((x, hidden), dim=-1)
        return self.layer(x)
