import torch
from torch import sigmoid, tanh
import torch.nn as nn

from rnns.helpers import RNNGate


class GRUcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        # Size of the H and C states
        self.hidden_size = hidden_size
        concated_input = input_size + hidden_size
        self.reset_gate = RNNGate(concated_input, hidden_size, sigmoid)
        self.update_gate = RNNGate(concated_input, hidden_size, sigmoid)
        self.tanh3 = RNNGate(concated_input, hidden_size, tanh)

    def _init_hidden(self, batch_size):
        """
        Return initial h, c states. Which are all zeros
        """
        return torch.zeros((batch_size, self.hidden_size), dtype=torch.float)

    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size = x.size(0)
            h = self._init_hidden(batch_size)
        else:
            h = hidden

        x_cat_1 = torch.cat((h, x), dim=-1)
        r_t = self.reset_gate(x_cat_1)
        z_t = self.update_gate(x_cat_1)

        x_cat_2 = torch.cat((r_t * h, x), dim=-1)
        h_tilde = self.tanh3(x_cat_2)

        forget = (1 - z_t) * h
        new_info = z_t * h_tilde
        h = forget + new_info

        return h, h
