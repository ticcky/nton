import unittest

import numpy as np
from nn.dbmap import DBMap
from nn.utils import check_finite_differences


class TestDBMap(unittest.TestCase):
    def test_forward(self):
        map = DBMap([0, 5, 9])

        x = np.abs(np.random.randn(10))
        x /= x.sum()
        db1 = np.abs(np.random.randn(10))
        db1 /= db1.sum()
        db2 = np.abs(np.random.randn(10))
        db2 /= db2.sum()
        db3 = np.abs(np.random.randn(10))
        db3 /= db3.sum()

        ((y, ), aux) = map.forward((x , db1, db2, db3))

        mass = x[0] + x[5] + x[9]
        y_true = x[0] * db1 + x[5] * db2 + x[9] * db3 + (1 - mass) * x

        self.assertTrue(np.allclose(y, y_true))

    def test_backward(self):
        map = DBMap([0, 5, 9])

        def gen():
            res = np.abs(np.random.randn(10))
            return res / res.sum()

        def gen_input():
            return (gen(), gen(), gen(), gen(), )

        self.assertTrue(
            check_finite_differences(
                map.forward,
                map.backward,
                gen_input_fn=gen_input,
                aux_only=True
            )
        )





if __name__ == '__main__':
    unittest.main()
