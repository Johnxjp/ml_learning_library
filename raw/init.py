import numpy as np

from raw.constants import Array


def kaiming_normal(n_in: int, n_out: int, a: float = 0) -> Array:
    """
    The scaling factor should change for different activations. For ReLU,
    use a = 0, for linear a = 1 and for leaky_relu, use the same value as
    the gradient of the negative slope.
    """
    scaling_factor = 2 / (1 + a ** 2)
    std = np.sqrt(scaling_factor / n_in)
    return np.random.normal(0, std, size=(n_in, n_out))


def kaiming_uniform(n_in: int, n_out: int, a: float = 0) -> Array:
    """
    The scaling factor should change for different activations. For ReLU,
    use a = 0, for linear (sigmoid?),  a = 1 and for leaky_relu,
    use the same value as the gradient of the negative slope.
    """
    scaling_factor = 6 / (1 + a ** 2)
    limit = np.sqrt(scaling_factor / n_in)
    return np.random.uniform(-limit, limit, size=(n_in, n_out))


def xavier_normal(n_in: int, n_out: int) -> Array:
    std = np.sqrt(2 / (n_in + n_out))
    return np.random.normal(0, std, size=(n_in, n_out))


def xavier_uniform(n_in: int, n_out: int) -> Array:
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size=(n_in, n_out))
