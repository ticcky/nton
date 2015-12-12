import unittest
import numpy as np

from sum import Sum
from utils import check_finite_differences


class TestSum(unittest.TestCase):
    def test(self):
        def gen():
            return (np.random.randn(20), )

        self.assertTrue(
            check_finite_differences(
                Sum.forward,
                Sum.backward,
                gen_input_fn=gen
            )
        )


if __name__ == '__main__':
    unittest.main()
