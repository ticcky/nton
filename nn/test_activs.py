from unittest import TestCase, main
import numpy as np

from activs import Tanh
from utils import check_finite_differences


class TestTanh(TestCase):
    def test_forward(self):
        x = np.array([[1, 1, 1], [0, 1, 2]])
        ((res, ), aux, ) = Tanh.forward((x, ))

        self.assertTrue(np.allclose(res[0], np.tanh(x[0])))
        self.assertTrue(np.allclose(res[1], np.tanh(x[1])))

    def test_backward(self):
        self.assertTrue(
            check_finite_differences(
                Tanh.forward,
                Tanh.backward,
                gen_input_fn=lambda: (np.random.randn(7, 3), )
            )
        )


if __name__ == '__main__':
    main()