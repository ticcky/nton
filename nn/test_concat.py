import unittest
import numpy as np

from concat import Concat


class TestConcat(unittest.TestCase):
    def test(self):
        shp = [5, 7, 9, 13, 3]
        inputs = tuple(np.random.randn(shp[i])for i in range(5))

        ((y, ), aux) = Concat.forward(inputs)
        dinputs = Concat.backward(aux, (np.random.randn(*y.shape), ))

        self.assertEqual(len(inputs), len(dinputs))

        for x, dx in zip(inputs, dinputs):
            self.assertEqual(len(x), len(dx))


if __name__ == '__main__':
    unittest.main()
