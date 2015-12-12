import numpy as np
from unittest import TestCase, main

from nn.attention import Attention
from nn.utils import TestParamGradInLayer, check_finite_differences
from nn.vars import Vars


class TestAttention(TestCase):
    def test_forward(self):
        att = Attention(n_hidden=5)

        h_out = np.random.randn(11, 5)
        g_t = np.random.randn(5)
        emb_in = np.random.randn(11, 13)  # Input emb size 13.

        ((query, ), aux) = att.forward((h_out, g_t, emb_in))
        self.assertEqual(query.shape, (13, ))

    def test_backward(self):
        att = Attention(n_hidden=5)

        def gen_input():
            h_out = np.random.randn(11, 5)
            g_t = np.random.randn(5)
            emb_in = np.random.randn(11, 13)  # Input emb size 13.

            return (h_out, g_t, emb_in, )

        self.assertTrue(
            check_finite_differences(
                att.forward,
                att.backward,
                gen_input_fn=gen_input,
                test_inputs=(0, 1, 2),
                aux_only=True
            )
        )
        TestParamGradInLayer.check_layers_params(
            att,
            gen_input(),
            self.assertTrue
        )




if __name__ == "__main__":
    main()