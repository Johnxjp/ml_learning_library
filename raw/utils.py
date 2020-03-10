import numpy as np

from raw.constants import Array


def bmm_outer(A: Array, B: Array) -> Array:
    """
    Computes the outer product for two matrices which are of rank-d and the
    first element is the batch dimension.
    """
    A = np.expand_dims(A, axis=-1)
    B = np.expand_dims(B, axis=1)
    return A @ B


def vector_jacobian_ce(dc_ds: Array, softmax_derivative: Array) -> Array:
    """
    dc_ds is the derivative of the cost function wrt to the softmax.
    Both are batch vectors and should have the following dimensions

    dc_ds: B x S
    softmax_derivative: B x S x S
    """
    return np.expand_dims(dc_ds, axis=1) @ softmax_derivative
