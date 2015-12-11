import unittest
import numpy as np

from tracker import Tracker


class TestTracker(unittest.TestCase):
    def test(self):
        tr = np.random.randn(10)
        slu = np.random.randn(10)
        h_t = np.random.randn(5)
        s = np.random.randn(7)

        tracker = Tracker(5, 7)

        ((y, ), aux) = tracker.forward((tr, slu, h_t, s, ))
        (dtr, dslu, dh_t, ds, ) = tracker.backward(aux, (np.random.randn(*y.shape), ))

        self.assertEqual(len(dtr), len(tr))
        self.assertEqual(len(dslu), len(slu))
        self.assertEqual(len(dh_t), len(h_t))
        self.assertEqual(len(ds), len(s))


if __name__ == '__main__':
    unittest.main()
