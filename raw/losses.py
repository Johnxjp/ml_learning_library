import numpy as np

from raw.constants import Array
from raw.activations import softmax


def binary_cross_entropy(inputs: Array, targets: Array) -> Array:
    """
    Targets will be one hot vectors.

    Inputs shape = (N, *)
    Targets shape = (N, *)
    """
    positive = lambda x: -np.log(x)
    negative = lambda x: -np.log(1 - x)
    return np.where(targets == 1, positive(inputs), negative(inputs))


def d_binary_cross_entropy(inputs: Array, targets: Array) -> Array:
    positive = lambda x: -1 / x
    negative = lambda x: 1 / (1 - x)
    return np.where(targets == 1, positive(inputs), negative(inputs))


def cross_entropy(inputs: Array, targets: Array) -> Array:
    return -1 * targets * np.log(inputs)


def cross_entropy_with_logits(inputs: Array, targets: Array) -> Array:
    inputs = softmax(inputs)
    return -1 * targets * np.log(inputs)


def d_cross_entropy(inputs: Array, targets: Array) -> Array:
    return -1 * targets / inputs


def d_cross_entropy_with_logits(inputs: Array, targets: Array) -> Array:
    inputs = softmax(inputs)
    return -1 * targets / inputs
