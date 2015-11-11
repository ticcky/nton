import numpy as np

from base import Block
from vars import Vars
from softmax import Softmax
from activs import Tanh
from linear import Dot


class Switch(Block):
    @classmethod
    def forward(self, (p1, in1, in2)):
        res = p1 * in1 + (1 - p1) * in2

        aux = Vars(
            p1=p1,
            in1=in1,
            in2=in2
        )

        return ((res, ), aux)

    @classmethod
    def backward(self, aux, (dres, )):
        p1 = aux['p1']
        in1 = aux['in1']
        in2 = aux['in2']

        din1 = p1 * dres
        din2 = (1 - p1) * dres

        dp1 = np.dot(dres, in1)
        dp1 -= np.dot(dres, in2)

        return (np.array([dp1]), din1, din2)
