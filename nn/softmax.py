import numpy as np

from base import Block
from vars import Vars


class Softmax(Block):
    """Compute softmax of the input."""
    def __init__(self):
        pass

    @classmethod
    def forward(self, (x, )):
        xmax = x.max(axis=x.ndim - 1, keepdims=True)
        res = np.exp(x - xmax)
        ndx = ((slice(None), ) * (len(x.shape) - 1)) + (None, )
        res = res / np.sum(res, axis=len(x.shape) - 1)[ndx]

        aux = Vars(y=res)

        return ((res, ), aux, )

    @classmethod
    def backward(self, (x, ), aux, (dy, )):
        y = aux['y']

        res = y * dy
        s = res.sum(axis=res.ndim - 1, keepdims=True)
        res -= y * s

        return (res, )