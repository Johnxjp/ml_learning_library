from typing import Union
import numpy as np

from raw.constants import Array
from raw.utils import bmm_outer


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
    out = np.where(x <= 0, 0.0, 1.0)
    if isinstance(x, int):
        return int(out.item())
    if isinstance(x, float):
        return out.item()
    return out


def softmax(x: Union[Array, int, float]) -> Union[Array, int, float]:
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def d_softmax(x: Union[Array, int, float]) -> Union[Array, int, float]:
    if isinstance(x, int) or isinstance(x, float):
        return 1

    out = bmm_outer(x, x)
    diag = np.apply_along_axis(np.diag, 1, x)
    return out - diag
