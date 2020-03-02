import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import RNNGate


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        # Size of the H and C states
        self.hidden_size = hidden_size
        concated_input = input_size + hidden_size
        self.sigmoid1 = RNNGate(concated_input, hidden_size, F.sigmoid)
        self.sigmoid2 = RNNGate(concated_input, hidden_size, F.sigmoid)
        self.sigmoid3 = RNNGate(concated_input, hidden_size, F.sigmoid)
        self.tanh4 = RNNGate(concated_input, hidden_size, F.tanh)

    def _init_hidden(self, batch_size):
        """
        Return initial h, c states. Which are all zeros
        """
        return (
            torch.zeros((batch_size, self.hidden_size), dtype=torch.float),
            torch.zeros((batch_size, self.hidden_size), dtype=torch.float),
        )

    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size = x.size(0)
            h, c = self._init_hidden(batch_size)
        else:
            h, c = hidden

        x = torch.cat((h, x), dim=-1)
        # Forget
        forget = self.sigmoid1(x)
        c *= forget

        # Update
        i_t = self.sigmoid2(x)
        c_t = self.tanh4(x)
        new_info = i_t * c_t
        c += new_info

        # Filter output
        c_filter = F.tanh(c)
        h = self.sigmoid3(x) * c_filter

        return h, (h, c)
