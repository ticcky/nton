import numpy as np
from unittest import TestCase, main

from lstm import LSTM
from utils import TestParamGradInLayer, check_finite_differences
from vars import Vars


class TestLSTM(TestCase):
    def test_forward(self):
        lstm = LSTM(n_in=5, n_out=7)

        input = np.random.randn(11, 3, 5)
        ((h, ), aux_lstm) = lstm.forward((input, None, None ))

        self.assertEqual(h.shape, (11, 3, 7))

    def test_backward(self):
        lstm = LSTM(n_in=10, n_out=4)

        x = np.random.randn(5, 3, 10)
        c0 = np.random.randn(3, 4)
        h0 = np.random.randn(3, 4)
        inp = (x, c0, h0, )

        params_shape = lstm.params['WLSTM'].shape

        checker = TestParamGradInLayer(lstm, 'WLSTM', layer_input=inp)
        check = check_finite_differences(
            checker.forward,
            checker.backward,
            gen_input_fn=lambda: (np.random.randn(*params_shape), )
        )
        self.assertTrue(check)


if __name__ == "__main__":
    main()