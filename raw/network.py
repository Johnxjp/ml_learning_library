"""
Raw implementation of neural network
"""
from typing import List

import numpy as np

from raw import init
from raw.constants import Array
from raw.activations import sigmoid, softmax


class Linear:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights = init.kaiming_normal(input_size, output_size, a=0)
        self.bias = np.zeros(output_size)

    def forward(self, x: Array) -> Array:
        return x @ self.weights + self.bias

    def __call__(self, x: Array) -> Array:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"Linear - weights: {self.weights.shape}, bias: {self.bias.shape}"
        )


class LinearANN:
    def __init__(
        self, input_size: int, hidden_layers: List[int], output_size: int
    ) -> None:
        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.layers = []
        for inp, out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(Linear(inp, out))

        self.forward_activations = []

    def forward(self, x: Array) -> Array:
        self.forward_activations = [x]
        for layer in self.layers[:-1]:
            x = sigmoid(layer(x))
            self.forward_activations.append(x)

        out = softmax(self.layers[-1](x))
        return out

    def __call__(self, x: Array) -> Array:
        return self.forward(x)

    def _summary(self) -> str:
        string_rep = "LinearANN {\n"
        for layer in self.layers:
            string_rep += "  " + str(layer) + "\n"
        string_rep += "}"
        return string_rep

    def __repr__(self) -> str:
        return self._summary()
