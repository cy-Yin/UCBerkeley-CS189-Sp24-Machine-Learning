import numpy as np
from numpy.linalg import norm
from typing import Callable


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def integers_to_one_hot(integer_vector, max_val=None):
    integer_vector = np.squeeze(integer_vector)
    if max_val == None:
        max_val = np.max(integer_vector)
    one_hot = np.zeros((integer_vector.shape[0], max_val + 1))
    for i, integer in enumerate(integer_vector):
        one_hot[i, integer] = 1.0
    return one_hot


def center(X, axis=0):
    return X - np.mean(X, axis=axis)


def normalize(X, axis=0, max_val=None):
    X -= np.min(X, axis=axis)
    if max_val is None:
        X /= np.max(X, axis=axis)
    else:
        X /= max_val
    return X


def standardize(X, axis=0):
    mean = np.mean(X, axis=axis)
    std = np.std(X, axis=axis)
    X -= mean
    X /= std + 1e-10
    return X


def check_gradients(
    fn: Callable[[np.ndarray], np.ndarray],
    grad: np.ndarray,
    x: np.ndarray,
    dLdf: np.ndarray,
    h: float = 1e-6,
) -> float:
    """Performs numerical gradient checking by numerically approximating
    the gradient using a two-sided finite difference.

    For each position in `x`, this function computes the numerical gradient as:
        numgrad = fn(x + h) - fn(x - h)
                  ---------------------
                            2h

    Next, we use the chain rule to compute the derivative of the input of `fn`
    with respect to the loss:
        numgrad = numgrad @ dLdf

    The function then returns the relative difference between the gradients:
        ||numgrad - grad||/||numgrad + grad||

    Parameters
    ----------
    fn       function whose gradients are being computed
    grad     supposed to be the gradient of `fn` at `x`
    x        point around which we want to calculate gradients
    dLdf     derivative of
    h        a small number (used as described above)

    Returns
    -------
    relative difference between the numerical and analytical gradients
    """
    # ONLY WORKS WITH FLOAT VECTORS
    if x.dtype != np.float32 and x.dtype != np.float64:
        raise TypeError(f"`x` must be a float vector but was {x.dtype}")

    # initialize the numerical gradient variable
    numgrad = np.zeros_like(x)

    # compute the numerical gradient for each position in `x`
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        pos = fn(x).copy()
        x[ix] = oldval - h
        neg = fn(x).copy()
        x[ix] = oldval

        # compute the derivative, also apply the chain rule
        numgrad[ix] = np.sum((pos - neg) * dLdf) / (2 * h)
        it.iternext()

    return norm(numgrad - grad) / norm(numgrad + grad)
