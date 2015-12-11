import unittest
import numpy as np

from concat import Concat

from nn.utils import check_finite_differences


class TestConcat(unittest.TestCase):
    def test(self):
        shp = [5, 7, 9, 13, 3]

        def gen_input():
            return tuple(np.random.randn(shp[i])for i in range(5))

        inputs = gen_input()

        ((y, ), aux) = Concat.forward(inputs)
        dinputs = Concat.backward(aux, (np.random.randn(*y.shape), ))

        self.assertEqual(len(inputs), len(dinputs))

        for x, dx in zip(inputs, dinputs):
            self.assertEqual(len(x), len(dx))

        self.assertTrue(check_finite_differences(
            Concat.forward,
            Concat.backward,
            gen_input_fn=gen_input,
            aux_only=True
        ))


if __name__ == '__main__':
    unittest.main()
