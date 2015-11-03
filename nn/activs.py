import numpy as np

from base import Block
from vars import Vars


class Tanh(Block):
    def __init__(self):
        pass

    @classmethod
    def forward(self, (x, )):
        y = np.tanh(x)
        aux = Vars(y=y)

        return ((y, ), aux)

    @classmethod
    def backward(self, (x, ), aux, (dy, )):
        y = aux['y']
        res = (1 - y * y) * dy

        return (res, )