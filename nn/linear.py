from base import Block, ParametrizedBlock
from inits import Normal
from vars import Vars

import numpy as np


class Identity(Block):
    def forward(self, x):
        return x

    def backward(self, x, y, dy):
        return dy


class LinearLayer(ParametrizedBlock):
    """Affine transformation."""
    def __init__(self, n_in, n_out, init_w=Normal(), init_b=Normal()):
        W = init_w((n_in, n_out))
        b = init_b((n_out, ))
        params = Vars(W=W, b=b)

        dW = np.zeros_like(W)
        db = np.zeros_like(b)
        grads = Vars(W=dW, b=db)

        self.parametrize(params, grads)

    def forward(self, (x, )):
        W = self.params['W']
        b = self.params['b']

        y = np.dot(x, W) + b

        aux = Vars(
            y=y
        )

        return ((y, ), aux)

    def backward(self, (x, ), aux, (dy, )):
        y = aux['y']

        W = self.params['W']
        self.grad_accum(x, y, dy)
        res = np.dot(dy, W.T)

        return (res, )

    def grad_accum(self, x, y, dy):
        x = x.reshape((-1, x.shape[-1]))
        dy = dy.reshape((-1, dy.shape[-1]))

        dW = np.dot(x.T, dy)
        db = dy.sum(axis=0)

        self.grads['W'] += dW
        self.grads['b'] += db




class Dot(ParametrizedBlock):
    """Dot product."""
    @classmethod
    def forward(self, (A, B)):
        if A.ndim == 1:
            A = A[np.newaxis, :]
        if B.ndim == 1:
            B = B[:, np.newaxis]

        return ((np.dot(A, B), ), None)

    @classmethod
    def backward(self, (A, B, ), aux, (dy, )):
        if A.ndim == 1:
            A = A[:, np.newaxis]
        if B.ndim == 1:
            B = B[:, np.newaxis]

        dA = np.dot(dy, B.T)
        dB = np.dot(A.T, dy)

        return (dA, dB)
