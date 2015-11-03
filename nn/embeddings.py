import numpy as np

from base import ParametrizedBlock
from inits import Normal
from vars import Vars


class Embeddings(ParametrizedBlock):
    """Embedding layer.
    Takes a tensor of integres as input and returns a tensor one order greater
    with the last dimension being n_dims where the integer ids are mapped through
    parameter matrix W to their embeddings."""
    def __init__(self, n_tokens, n_dims, init_fn=Normal()):
        self.n_tokens = n_tokens
        self.n_dims = n_dims

        W = init_fn((n_tokens, n_dims))
        params = Vars(W=W)

        dW = np.zeros_like(W)
        grads = Vars(W=dW)

        self.parametrize(params, grads)

    def forward(self, (x, )):
        """Map input indicies to embedding vectors."""
        W = self.params['W']

        assert x.ndim == 1, 'Cannot embed non-vector arrays.'

        y = W[x]

        return ((y, ), None, )

    def backward(self, inputs, aux, grads):
        self.grad_accum(inputs, aux, grads)

        return (None, )  # Non-differentiable.

    def grad_accum(self, inputs, aux, grads):
        dW = self.grads['W']

        x = inputs[0]
        dy = grads[0]

        for i in range(len(x)):
            dW[x[i]] += dy[i]