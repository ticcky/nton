import unittest
import numpy as np

from manager import Manager
from nn.utils import check_finite_differences, TestParamGradInLayer


class TestManager(unittest.TestCase):
    def test(self):
        def gen_input():
            s = np.random.randn(10)
            h_t = np.random.randn(5)
            db_count = np.random.randn(1)

            return (s, h_t, db_count, )

        test_input = (s, h_t, db_count) = gen_input()

        mgr = Manager(5, 10, 1, hidden_size=128)

        ((s_prime, ), aux) = mgr.forward((s, h_t, db_count, ))
        (ds, dh_t, ddb_count, ) = mgr.backward(aux, (np.random.randn(*s_prime.shape), ))

        self.assertEqual(len(ds), len(s))
        self.assertEqual(len(dh_t), len(h_t))
        self.assertEqual(len(ddb_count), len(db_count))


        self.assertTrue(check_finite_differences(
            mgr.forward,
            mgr.backward,
            gen_input_fn=gen_input,
            aux_only=True,
            test_inputs=(0, 1, 2, )
        ))

        for param_name in mgr.params.var_names:
            layer = TestParamGradInLayer(mgr, param_name, test_input)
            self.assertTrue(check_finite_differences(
                layer.forward,
                layer.backward,
                gen_input_fn=layer.gen_input
            ), 'Gradient check for param: %s' % param_name)




if __name__ == '__main__':
    unittest.main()
