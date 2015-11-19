import numpy as np

from nn import Block, Vars


class SeqLoss(Block):
    #epsilon = 1e-7

    @classmethod
    def forward(self, (y_hat, y_true)):
        assert len(y_hat) == len(y_true), 'Outputs do not match.'

        res = 0.0
        grad = np.zeros_like(y_hat)
        for i, (yi_hat, yi_true) in enumerate(zip(y_hat, y_true)):
            if yi_true == -1:
                continue

            res += - np.log(yi_hat[yi_true])
            grad[i, yi_true] = - 1.0 / (yi_hat[yi_true])

        res /= len(y_hat)
        grad /= len(y_hat)

        aux = Vars(
            y_hat=y_hat,
            y_true=y_true,
            grad=grad
        )

        return ((res, ), aux)

    @classmethod
    def backward(self, aux, dy):
        return (aux['grad'] * dy, )



