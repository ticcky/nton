import unittest
import numpy as np

from data_calc import DataCalc
from db import DB
from nn import OneHot
from nton import NTON
from nn.utils import check_finite_differences


class TestNTON(unittest.TestCase):
    def test_forward_backward(self):
        calc = DataCalc()
        db = DB(calc.get_db(), calc.get_vocab())
        emb = OneHot(n_tokens=len(db.vocab))

        nton = NTON(
            n_tokens=len(db.vocab),
            db=db,
            emb=emb,
            n_cells=5
        )

        ((E, ), _) = emb.forward(([1, 2, 3], ))
        ((dec_sym, ), _) = emb.forward(([db.vocab['[EOS]']], ))

        ((Y, y), Y_aux) = nton.forward((E, dec_sym))

        self.assertEqual(len(y), len(Y))
        self.assertEqual(len(y), nton.max_gen)

    def test_backward(self):
        calc = DataCalc(max_num=2, n_words=5)
        db = DB(calc.get_db(), calc.get_vocab())

        emb = OneHot(n_tokens=len(db.vocab))

        nton = NTON(
            n_tokens=len(db.vocab),
            db=db,
            emb=emb,
            n_cells=5
        )
        ((dec_sym, ), _) = emb.forward(([db.vocab['[EOS]']], ))

        def gen_input():
            ((E, ), _) = emb.forward((np.random.randint(1, len(db.vocab), (3, )), ))

            return (E, dec_sym, )

        check = check_finite_differences(
            nton.forward,
            nton.backward,
            gen_input_fn=gen_input,
            aux_only=True
        )
        self.assertTrue(check)


if __name__ == '__main__':
    unittest.main()
