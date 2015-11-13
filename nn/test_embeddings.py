import numpy as np
from unittest import TestCase, main

from nn.embeddings import Embeddings, OneHot
from nn.utils import TestParamGradInLayer, check_finite_differences
from nn.vars import Vars


class TestOneHot(TestCase):
    def test_forward(self):
        hot = OneHot(n_tokens=7)

        ((y, ), aux) = hot.forward((np.array([1, 6, 3]), ))



class TestEmbeddings(TestCase):
    def test_forward(self):
        emb = Embeddings(n_tokens=10, n_dims=100)
        ((y_emb, ), aux_emb) = emb.forward((np.array([0, 1, 9]), ))

        # Check that correct embeddings are pulled and given in the right shape.
        self.assertEqual(y_emb.shape, (3, 100, ))
        self.assertTrue(np.allclose(y_emb[2], emb.params['W'][9]))

    def test_backward(self):
        emb = Embeddings(n_tokens=10, n_dims=100)

        inp = (np.array([0, 1, 9]), )

        checker = TestParamGradInLayer(emb, 'W', layer_input=inp)
        check = check_finite_differences(
            checker.forward,
            checker.backward,
            gen_input_fn=lambda: (np.random.randn(*emb.params['W'].shape), ),
            aux_only=True
        )
        self.assertTrue(check)


if __name__ == "__main__":
    main()