from unittest import TestCase, main
import numpy as np

from nn.activs import Tanh, Sigmoid, ReLU, Normalize
from nn.utils import check_finite_differences


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
                gen_input_fn=lambda: (np.random.randn(7, 3), ),
                aux_only=True
            )
        )


class TestSigmoid(TestCase):
    def test_forward(self):
        x = np.array([[1, 1, 1], [0, 1, 2]])
        ((res, ), aux, ) = Sigmoid.forward((x, ))

        self.assertTrue(np.allclose(res[0], 0.5 * (1 + np.tanh(0.5 * x[0]))))
        self.assertTrue(np.allclose(res[1], 0.5 * (1 + np.tanh(0.5 * x[1]))))

    def test_backward(self):
        self.assertTrue(
            check_finite_differences(
                Sigmoid.forward,
                Sigmoid.backward,
                gen_input_fn=lambda: (np.random.randn(7, 3), ),
                aux_only=True
            )
        )


class TestReLU(TestCase):
    def test_forward(self):
        x = np.array([[1, -1, -10], [0, 1, 20]])
        ((res, ), aux, ) = ReLU.forward((x, ))

        self.assertTrue(np.allclose(res, [[1, 0, 0], [0, 1, 20]]))

    def test_backward(self):
        self.assertTrue(
            check_finite_differences(
                ReLU.forward,
                ReLU.backward,
                gen_input_fn=lambda: (np.random.randn(7, 3), ),
                aux_only=True
            )
        )


class TestNormalize(TestCase):
    def test_forward(self):
        x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        ((res, ), aux, ) = Normalize.forward((x, ))

        self.assertTrue(np.allclose(res, [0, 0.1, 0.2, 0.3, 0.4]))

    def test_backward(self):
        self.assertTrue(
            check_finite_differences(
                Normalize.forward,
                Normalize.backward,
                gen_input_fn=lambda: (np.abs(np.random.randn(7)), ),
                aux_only=True
            )
        )



if __name__ == '__main__':
    main()