import unittest
import numpy as np

from manager import Manager


class TestManager(unittest.TestCase):
    def test(self):
        s = np.random.randn(10)
        h_t = np.random.randn(5)
        db_count = np.random.randn(1)

        mgr = Manager(5, 10, 1, hidden_size=128)

        ((s_prime, ), aux) = mgr.forward((s, h_t, db_count, ))
        (ds, dh_t, ddb_count, ) = mgr.backward(aux, (np.random.randn(*s_prime.shape), ))

        self.assertEqual(len(ds), len(s))
        self.assertEqual(len(dh_t), len(h_t))
        self.assertEqual(len(ddb_count), len(db_count))


if __name__ == '__main__':
    unittest.main()
