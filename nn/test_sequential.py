from unittest import TestCase, main
import numpy as np

from utils import check_finite_differences
from sequential import Sequential
from inits import Constant, Eye
from lstm import LSTM
from softmax import Softmax
from linear import LinearLayer

class TestSequential(TestCase):
    def test_forward(self):
        seq = Sequential([
            LinearLayer(n_in=5, n_out=2, init_w=Eye(), init_b=Constant(0.0)),
            Softmax()
        ])

        ((y, ), _) = seq.forward(np.array([[1, 0, 0, 0, 0]]))
        self.assertTrue(np.allclose(y, [0.73105858, 0.26894142]))

        ((y, ), _) = seq.forward(np.array([[0, 0, 0, 0, 0]]))
        self.assertTrue(np.allclose(y, [0.5, 0.5]))

    def test_backward(self):
        seq = Sequential([
            LinearLayer(n_in=5, n_out=2, init_w=Eye(), init_b=Constant(0.0)),
            Softmax()
        ])

        check = check_finite_differences(
            fwd_fn=seq.forward,
            bwd_fn=seq.backward,
            gen_input_fn=lambda: (np.random.randn(3, 5), )
        )
        self.assertTrue(check)


if __name__ == "__main__":
    main()
