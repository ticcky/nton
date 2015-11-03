import numpy as np
from unittest import TestCase, main

from attention import Attention
from utils import TestParamGradInLayer, check_finite_differences
from vars import Vars


class TestAttention(TestCase):
    def test_forward(self):
        att = Attention(n_hidden=5)

        h_out = np.random.randn(11, 5)
        g_t = np.random.randn(5)
        emb_in = np.random.randn(11, 13)  # Input emb size 13.

        ((query, ), aux) = att.forward((h_out, g_t, emb_in))
        print query.shape


        #self.assertEqual(h.shape, (11, 3, 7))

    def test_backward(self):
        att = Attention(n_hidden=5)

        def gen_input():
            h_out = np.random.randn(11, 5)
            g_t = np.random.randn(5)
            emb_in = np.random.randn(11, 13)  # Input emb size 13.

            return (h_out, g_t, emb_in, )

        check = check_finite_differences(
            att.forward,
            att.backward,
            gen_input_fn=gen_input
        )
        print check

        return

        h_out = np.random.randn(11, 5)
        g_t = np.random.randn(5)
        emb_in = np.random.randn(11, 13)  # Input emb size 13.

        ((query, ), aux) = att.forward((h_out, g_t, emb_in))

        dquery = np.random.randn(13)

        att.backward((h_out, g_t, emb_in, ), aux, (dquery, ))

        # lstm = LSTM(n_in=10, n_out=4)
        #
        # x = np.random.randn(5, 3, 10)
        # c0 = np.random.randn(3, 4)
        # h0 = np.random.randn(3, 4)
        # inp = (x, c0, h0, )
        #
        # params_shape = lstm.params['WLSTM'].shape
        #


        # self.assertTrue(check)
        pass


if __name__ == "__main__":
    main()