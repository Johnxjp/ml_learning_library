"""
Raw implementation of neural network
"""
from typing import List

import numpy as np

from raw import init
from raw.constants import Array
from raw.activations import sigmoid, d_sigmoid


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

    def _reset_activations(self):
        self.forward_activations = []

    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            z = layer(x)
            self.forward_activations.append((x, z))
            x = sigmoid(z)

        return x

    def __call__(self, x: Array) -> Array:
        return self.forward(x)

    def backward(self, loss_derivative: Array, lr: float = 0.01) -> None:
        """Computes gradients and updates params based on standard SGD"""

        # Remember this is batch!
        activations = reversed(self.forward_activations)
        layers = reversed(self.layers)

        dx = loss_derivative
        for (a_in, z), layer in zip(activations, layers):
            dx *= d_sigmoid(z)
            a_in = np.expand_dims(a_in, axis=-1)
            dx_ = np.expand_dims(dx, axis=1)

            dw = lr * (a_in @ dx_)
            db = lr * dx

            layer.weights -= np.mean(dw, axis=0)
            layer.bias -= np.mean(db, axis=0)

            # Reshape
            dx = dx @ layer.weights.T

        # Clear the activations for next time
        self._reset_activations()

    def _summary(self) -> str:
        string_rep = "LinearANN {\n"
        for layer in self.layers:
            string_rep += "  " + str(layer) + "\n"
        string_rep += "}"
        return string_rep

    def __repr__(self) -> str:
        return self._summary()
