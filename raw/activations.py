from typing import Union
import numpy as np

from raw.constants import Array


def sigmoid(x: Union[Array, int, float]) -> Union[Array, int, float]:
    z = np.exp(-x)
    return 1 / (1 + z)


def d_sigmoid(x: Union[Array, int, float]) -> Union[Array, int, float]:
    """
    Derivative with respect to z where z is sum (x @ w + b).
    """
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x: Union[Array, int, float]) -> Union[Array, int, float]:
    return np.maximum(0, x)


def d_relu(x: Union[Array, int, float]) -> Union[Array, int, float]:
    """
    Derivative with respect to z where z is sum (x @ w + b).
    """
    out = np.where(x <= 0, 0, 1)
    if isinstance(x, int) or isinstance(x, float):
        return out.item()


def softmax(x: Union[Array, int, float]) -> Union[Array, int, float]:
    return np.exp(x) / np.sum(np.exp(x))
