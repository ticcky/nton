import unittest
import numpy as np

from data_calc import DataCalc
from db import DB
from nn import OneHot
from nton import NTON
from nn.utils import check_finite_differences, TestParamGradInLayer


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

    def test_backward_gen(self):
        calc = DataCalc(max_num=5, n_words=50)
        db = DB(calc.get_db(), calc.get_vocab())
        n_words = len(db.vocab)

        emb = OneHot(n_tokens=len(db.vocab))


        nton = NTON(
            n_tokens=len(db.vocab),
            db=db,
            emb=emb,
            n_cells=5
        )
        nton.print_step = lambda *args, **kwargs: None
        shapes = [
            (n_words, ),
            (nton.n_cells,),
            (nton.n_cells,),
            (6, nton.n_cells),
            (6, n_words)
        ]
        check = check_finite_differences(
            nton.forward_gen_step,
            nton.backward_gen_step,
            gen_input_fn=lambda: tuple(np.random.randn(*shp) for shp in shapes),
            aux_only=True,
            n_times=100
        )
        self.assertTrue(check)


    def test_backward(self):
        calc = DataCalc(max_num=5, n_words=50)
        db = DB(calc.get_db(), calc.get_vocab())

        emb = OneHot(n_tokens=len(db.vocab))

        nton = NTON(
            n_tokens=len(db.vocab),
            db=db,
            emb=emb,
            n_cells=5
        )
        nton.max_gen = 2
        ((dec_sym, ), _) = emb.forward(([db.vocab['[EOS]']], ))

        def gen_input():
            ((E, ), _) = emb.forward((np.random.randint(1, len(db.vocab), (5, )), ))

            return (E, dec_sym, )

        # check = check_finite_differences(
        #     nton.forward,
        #     nton.backward,
        #     gen_input_fn=gen_input,
        #     aux_only=True
        # )
        # self.assertTrue(check)

        # ['att__Wh', 'att__Wy', 'att__w', 'in_rnn__WLSTM', 'out_rnn__WLSTM',
        # 'out_rnn_clf__00__W', 'out_rnn_clf__00__b', 'switch__00__W', 'switch__00__b']

        #for param_name in ['out_rnn_clf__00__W', 'out_rnn_clf__00__b']: #nton.params.names():
        for param_name in ['switch__00__W']: #, 'switch__00__b']: #nton.params.names():
            params_shape = nton.params[param_name].shape

            checker = TestParamGradInLayer(nton, param_name, layer_input=gen_input())
            check = check_finite_differences(
                checker.forward,
                checker.backward,
                gen_input_fn=lambda: (np.random.randn(*params_shape), ),
                aux_only=True,
                test_outputs=(0, ),
                n_times=100
            )
            self.assertTrue(check, msg='Failed check for: %s' % param_name)


if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
