import unittest

import numpy as np
import modules
from nn.utils import check_finite_differences


class TrackerSetTest(unittest.TestCase):
  def test_backward(self):
    trackers = modules.TrackerSet(5, 7, 3)

    def gen(size):
      return np.random.randn(size)


    def gen_input():
      return (gen(5), gen(7), gen(10), gen(10), gen(10), gen(10), gen(10), gen(10),)

    self.assertTrue(
        check_finite_differences(
            trackers.forward,
            trackers.backward,
            gen_input_fn=gen_input,
            aux_only=True
        )
    )


if __name__ == '__main__':
  unittest.main()
