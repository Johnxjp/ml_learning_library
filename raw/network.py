"""
Raw implementation of neural network
"""
from typing import List

import numpy as np

from raw import init
from raw.constants import Array
from raw.activations import relu, d_relu, softmax, d_softmax
from raw.utils import vector_jacobian_ce


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


class MultiClassNN:
    def __init__(
        self, input_size: int, hidden_layers: List[int], output_size: int
    ) -> None:
        if output_size < 2:
            raise ValueError(
                "Multiclass network should have at least two output neurons"
            )

        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.layers = []
        for inp, out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(Linear(inp, out))

        self.n_layers = len(self.layers)
        self.forward_activations = []

    def _reset_activations(self):
        self.forward_activations = []

    def forward(self, x: Array) -> Array:
        """
        Returns the logits before softmax
        """
        for n, layer in enumerate(self.layers):
            z = layer(x)
            self.forward_activations.append((x, z))
            # Do this to not apply relu to output layer
            if n < self.n_layers - 1:
                x = relu(z)

        return z

    def __call__(self, x: Array) -> Array:
        return self.forward(x)

    def backward(self, targets: Array, lr: float = 0.1) -> None:
        """Computes gradients and updates params based on standard SGD"""

        activations = reversed(self.forward_activations)
        layers = reversed(self.layers)

        for n, ((a_in, z), layer) in enumerate(zip(activations, layers)):
            if n == 0:
                # dx = vector_jacobian_ce(loss_derivative, d_softmax(z))
                # Equivalently, this just works out to be z - 1 along where y
                # is 1. z is the softmax values
                dx = softmax(z) - targets
            else:
                dx *= d_relu(z)

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
