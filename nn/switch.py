import numpy as np

from base import Block
from vars import Vars
from softmax import Softmax
from activs import Tanh
from linear import Dot


class Switch(Block):
    @classmethod
    def forward(self, inp):
        assert type(inp) == tuple

        p = inp[0]
        ins = inp[1:]

        assert len(p) == len(ins)

        res = np.zeros_like(ins[0])
        for p_i, ins_i in zip(p, ins):
            res += p_i * ins_i

        aux = Vars(
            p=p,
            ins=ins
        )

        return ((res, ), aux)

    @classmethod
    def backward(self, aux, (dres, )):
        assert type(dres) == np.ndarray

        p = aux['p']
        ins = aux['ins']

        dins = []
        for p_i in p:
            dins.append(p_i * dres)

        dp = np.zeros_like(p)
        for i, ins_i in enumerate(ins):
            dp[i] = np.dot(dres, ins_i)

        return (dp, ) + tuple(dins)
