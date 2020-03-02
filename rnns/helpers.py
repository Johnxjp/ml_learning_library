import torch.nn as nn


class RNNGate(nn.Module):
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        self.activation = activation
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.activation(self.layer(x))
