import numpy as np

from base import Block
from vars import Vars


class Tanh(Block):
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


class Sigmoid(Block):
    @classmethod
    def forward(self, (x, )):
        y = 0.5 * (1 + np.tanh(0.5 * x))

        aux = Vars(y=y)

        return ((y, ), aux)

    @classmethod
    def backward(self, (x, ), aux, (dy, )):
        y = aux['y']

        res = y * (1 - y) * dy

        return (res, )