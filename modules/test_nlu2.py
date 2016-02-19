import unittest
import numpy as np

from modules.nlu2 import NLU2
from nn.utils import check_finite_differences, TestParamGradInLayer



class TestNLU(unittest.TestCase):
    def test_forward(self):
        nlu = NLU2(10, 100, 3)

        E = np.random.randn(12, 100)

        ((h_n, slu1, slu2, slu3, ), aux) = nlu.forward((E, ))

        (dE, ) = nlu.backward(aux, (h_n, slu1, slu2, slu3, ) )

    def test_backward(self):
        nlu = NLU2(10, 11, 3)

        def gen():
            return (np.random.randn(12, 11), )

        self.assertTrue(
            check_finite_differences(
                nlu.forward,
                nlu.backward,
                gen_input_fn=gen,
                aux_only=True
            ),
            "Gradient check failed!"
        )
        TestParamGradInLayer.check_layers_params(
            nlu,
            gen(),
            self.assertTrue
        )


if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
