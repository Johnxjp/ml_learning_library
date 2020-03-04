"""
Raw implementation of neural network
"""

import numpy as np
from raw import init


class Linear:
    def __init__(self, input_size, output_size):
        self.weights = init.kaiming_normal(input_size, output_size, a=0)
        self.bias = np.zeros(output_size)

    def forward(self, x):
        return x @ self.weights + self.bias

    def __call__(self, x):
        return self.forward(x)
