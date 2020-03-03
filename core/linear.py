"""
Implementation of linear layer from scratch
"""

import torch
import torch.nn as nn


class LinearCore(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        # Normally, pytorch does swaps rows and columns
        self.weights = nn.Parameter(torch.Tensor(input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
        nn.init.kaiming_normal_(self.weights)

    def forward(self, x):
        return x @ self.weights + self.bias
