import numpy as np

from vars import Vars
from base import Block


class Concat(Block):
    @staticmethod
    def forward(inputs):
        assert type(inputs) == tuple
        assert all(len(x.shape) == 1 for x in inputs)

        y = np.concatenate(inputs)
        lengths = [x.shape[0] for x in inputs]

        return ((y,),  Vars(shapes=lengths))

    @staticmethod
    def backward(aux, (dy, )):
        shapes = aux['shapes']

        res = []
        curr = dy.copy()
        for l in shapes:
            res.append(curr[:l])
            curr = curr[l:]

        return tuple(res)


