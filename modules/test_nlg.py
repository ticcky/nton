import unittest
import numpy as np

from vocab import Vocab
from modules.nlg import NLG
from db_dist import DBDist
from nn.utils import check_finite_differences


class TestNLG(unittest.TestCase):
    def test(self):
        vocab_full = Vocab()
        vocab_full.add("<start>")
        vocab_full.add("#db1")
        vocab_full.add("#db2")
        vocab_full.add("#tr1")
        vocab_full.add("#tr2")
        vocab_full.add("#slu1")
        vocab_full.add("#slu2")
        vocab_full.add("ahoj")
        vocab_full.add("cau")
        vocab_full.add("nazdar")
        vocab_full.add("ahojky")
        vocab_full.add("nazdarek")
        vocab_full.add("valA1")
        vocab_full.add("valA2")
        vocab_full.add("valA3")
        vocab_full.add("valB1")
        vocab_full.add("valB2")
        vocab_full.add("valB3")

        start_token = DBDist.get_1hot_from("<start>", vocab_full)

        def gen_input():
            s_prime = np.random.randn(7)
            y_in = np.array([
                np.random.randn(*start_token.shape),
                np.random.randn(*start_token.shape),
                np.random.randn(*start_token.shape),
                np.random.randn(*start_token.shape),
            ])
            y_steps = 0

            other_inputs = [
                np.random.randn(*start_token.shape),
                np.random.randn(*start_token.shape),
                np.random.randn(*start_token.shape),
                np.random.randn(*start_token.shape),
                np.random.randn(*start_token.shape),
                np.random.randn(*start_token.shape),
            ]

            return (s_prime, y_in, y_steps, ) + tuple(other_inputs)

        mapping = [5, 6, 7, 8, 9, 10]

        inputs = gen_input()

        nlg = NLG(len(vocab_full), 7, mapping)
        ((O, ), aux) = nlg.forward(inputs)
        nlg.backward(aux, (np.random.randn(*O.shape), ))

        #self.assertEqual(O.shape, (8, 20))

        self.assertTrue(
            check_finite_differences(
                nlg.forward,
                nlg.backward,
                gen_input_fn=gen_input,
                aux_only=True,
                test_inputs=(0, 1, 3, 4, 5, 6, 7, 8, )
            )
        )



if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
