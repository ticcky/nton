import unittest
import numpy as np

from tracker import Tracker
from nn.utils import check_finite_differences, TestParamGradInLayer


class TestTracker(unittest.TestCase):
    def test(self):
        def gen_input():
            tr = np.random.randn(10)
            slu = np.random.randn(10)
            h_t = np.random.randn(5)
            s = np.random.randn(7)

            return (tr, slu, h_t, s, )

        test_input = (tr, slu, h_t, s) = gen_input()

        tracker = Tracker(5, 7)

        ((y, ), aux) = tracker.forward((tr, slu, h_t, s, ))
        (dtr, dslu, dh_t, ds, ) = tracker.backward(aux, (np.random.randn(*y.shape), ))

        self.assertEqual(len(dtr), len(tr))
        self.assertEqual(len(dslu), len(slu))
        self.assertEqual(len(dh_t), len(h_t))
        self.assertEqual(len(ds), len(s))

        self.assertTrue(check_finite_differences(
            tracker.forward,
            tracker.backward,
            gen_input_fn=gen_input,
            aux_only=True
        ))

        TestParamGradInLayer.check_layers_params(
            tracker,
            test_input,
            self.assertTrue
        )


if __name__ == '__main__':
    unittest.main()
