import numpy as np

from base import Block
from vars import Vars


class Sum(Block):
    @classmethod
    def forward(self, (x, )):
        y = np.sum(x, keepdims=True)

        return ((y, ), Vars(shp=x.shape))

    @classmethod
    def backward(self, aux, (dy, )):
        return (np.ones(*aux['shp']) * dy, )