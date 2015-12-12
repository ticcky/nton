import numpy as np

from base import ParametrizedBlock, Block
from inits import Normal
from vars import Vars


class OneHot(Block):
    def __init__(self, n_tokens):
        self.n_tokens = n_tokens

    def size(self):
        return self.n_tokens

    def forward(self, (x, )):
        res = np.zeros((len(x), self.n_tokens))
        res[range(len(x)), x] = 1

        return ((res, ), None)


class OneHotFromVocab(Block):
    def __init__(self, vocab):
        self.vocab = vocab

    def forward(self, (x, )):
        res = np.zeros((len(x), len(self.vocab), ))
        res[range(len(x)), tuple(self.vocab[i] for i in x)] = 1

        return ((res, ), None)


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

    def size(self):
        return self.n_dims

    def forward(self, (x, )):
        """Map input indicies to embedding vectors."""
        W = self.params['W']

        assert x.ndim == 1, 'Cannot embed non-vector arrays.'

        y = W[x]

        aux = Vars(
            x=x
        )

        return ((y, ), aux, )

    def backward(self, aux, grads):
        self.grad_accum(aux, grads)

        return (None, )  # Non-differentiable.

    def grad_accum(self, aux, grads):
        dW = self.grads['W']

        x = aux['x']
        dy = grads[0]

        for i in range(len(x)):
            dW[x[i]] += dy[i]