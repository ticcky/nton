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
    def backward(self, aux, (dy, )):
        y = aux['y']
        res = (1 - y * y) * dy

        return (res, )


class Sigmoid(Block):
    @classmethod
    def forward(self, (x, )):
        y = 0.5 * (1 + np.tanh(0.5 * x))

        aux = Vars(
            y=y
        )

        return ((y, ), aux)

    @classmethod
    def backward(self, aux, (dy, )):
        y = aux['y']

        res = y * (1 - y) * dy

        return (res, )

class ReLU(Block):
    @classmethod
    def forward(self, (x, )):
        y = (x > 0) * x

        aux = Vars(
            y=y
        )

        return ((y, ), aux)

    @classmethod
    def backward(self, aux, (dy, )):
        y = aux['y']

        res = (y > 0) * dy

        return (res, )

class Normalize(Block):
    @classmethod
    def forward(self, (x,)):
        xsum = np.sum(x)
        if xsum > 0:
          y = x / xsum
        else:
          y = x * 0
        aux = Vars(
            x=x,
            xsum=xsum
        )

        return ((y,), aux)

    @classmethod
    def backward(self, aux, (dy,)):
        xsumsq = (aux['xsum'] ** 2)
        if xsumsq > 0:
          base = np.sum((- dy * aux['x']) / xsumsq)
          res = base + dy / aux['xsum']
        else:
          res = np.zeros_like(dy)

        return (res, )


class Amplify(Block):
    @classmethod
    def forward(self, (x,)):
        y = np.zeros_like(x)
        max_index = np.argmax(x)
        if x[max_index] > 0:
          y[max_index] = 1

        aux = Vars(
            max_index=max_index
        )

        return ((y,), aux)

    @classmethod
    def backward(cls, aux, (dy,)):
        dx = np.zeros_like(dy)

        dx[aux['max_index']] = dy[aux['max_index']]

        return (dx,)