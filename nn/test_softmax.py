from unittest import TestCase, main
import numpy as np

from softmax import Softmax
from utils import check_finite_differences


class TestSoftmax(TestCase):
    def test_forward(self):
        softmax = Softmax()

        x = np.array([[1, 1, 1], [0, 1, 2]])
        ((res, ), aux, ) = softmax.forward((x, ))

        self.assertTrue(np.allclose(res[0], np.ones((3, )) / 3.0))
        self.assertTrue(np.allclose(res[1], [ 0.09003057, 0.24472847, 0.66524096]))

    def test_backward(self):
        softmax = Softmax()
        self.assertTrue(
            check_finite_differences(
                softmax.forward,
                softmax.backward,
                gen_input_fn=lambda: (np.random.randn(7, 3), )
            )
        )


if __name__ == '__main__':
    main()