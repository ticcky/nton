from unittest import TestCase, main
import numpy as np

from nn.linear import LinearLayer, Dot, LearnableInput
from nn.utils import check_finite_differences, TestParamGradInLayer


class TestLearnableInput(TestCase):
    def test_forward(self):
        lin = LearnableInput((10, 20))

        inp = np.random.randn(30, 10)
        ((res, ), _) = lin.forward(())

        self.assertTrue(res.shape == (10, 20))

    def test_backward(self):
        """Test gradient computation for inputs and all layer's parameters."""
        lin = LearnableInput((10, 20))

        inp = ()
        checker = TestParamGradInLayer(lin, 'W', inp)
        check = check_finite_differences(
            fwd_fn=checker.forward,
            bwd_fn=checker.backward,
            gen_input_fn=lambda: (np.random.randn(*lin.params['W'].shape), ),
            aux_only=True
        )
        self.assertTrue(check)


class TestLinearLayer(TestCase):
    def test_forward(self):
        lin = LinearLayer(n_in=10, n_out=5, init_w=lambda shape: np.eye(*shape), init_b=lambda shape: np.ones(shape) * 0.5)

        inp = np.random.randn(30, 10)

        ((res, ), _) = lin.forward((inp, ))

        self.assertTrue(np.allclose(inp[:, :5] + 0.5, res))

    def test_backward(self):
        """Test gradient computation for inputs and all layer's parameters."""
        linear = LinearLayer(n_in=10, n_out=5)

        check = check_finite_differences(
            fwd_fn=linear.forward,
            bwd_fn=linear.backward,
            gen_input_fn=lambda: (np.random.randn(30, 10), ),
            aux_only=True
        )
        self.assertTrue(check)

        inp = (np.random.randn(50, 30, 10), )
        checker = TestParamGradInLayer(linear, 'W', inp)
        check = check_finite_differences(
            fwd_fn=checker.forward,
            bwd_fn=checker.backward,
            gen_input_fn=lambda: (np.random.randn(*linear.params['W'].shape), ),
            aux_only=True
        )
        self.assertTrue(check)

        checker = TestParamGradInLayer(linear, 'b', inp)
        check = check_finite_differences(
            fwd_fn=checker.forward,
            bwd_fn=checker.backward,
            gen_input_fn=lambda: (np.random.randn(*linear.params['b'].shape), ),
            aux_only=True
        )
        self.assertTrue(check)

    def test_params(self):
        lin = LinearLayer(n_in=17, n_out=9)
        self.assertIsNotNone(lin.params['W'])
        self.assertIsNotNone(lin.params['b'])

        self.assertIsNotNone(lin.grads['W'])
        self.assertIsNotNone(lin.grads['b'])


class TestLinearTransform(TestCase):
    def test_forward(self):
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 7)

        ((AB, ), aux) = Dot.forward((A, B, ))
        (dA, dB) = Dot.backward(aux, (np.ones_like(AB) ,))

        A = np.random.randn(3, 4)
        B = np.random.randn(4)

        ((AB, ), aux) = Dot.forward((A, B, ))
        (dA, dB) = Dot.backward(aux, (np.ones_like(AB) ,))

        #self.assertTrue(np.allclose(A[:, :5] + 0.5, res))

    def test_backward(self):
        """Test gradient computation for inputs and all layer's parameters."""
        linear = LinearLayer(n_in=10, n_out=5)

        check = check_finite_differences(
            linear.forward,
            linear.backward,
            gen_input_fn=lambda: (np.random.randn(30, 10), ),
            aux_only=True
        )
        self.assertTrue(check)

        inp = (np.random.randn(50, 30, 10), )
        checker = TestParamGradInLayer(linear, 'W', layer_input=inp)
        check = check_finite_differences(
            checker.forward,
            checker.backward,
            gen_input_fn=lambda: (np.random.randn(*linear.params['W'].shape), ),
            aux_only=True
        )
        self.assertTrue(check)

        checker = TestParamGradInLayer(linear, 'b', layer_input=inp)
        check = check_finite_differences(
            checker.forward,
            checker.backward,
            gen_input_fn=lambda: (np.random.randn(*linear.params['b'].shape), ),
            aux_only=True
        )
        self.assertTrue(check)

    def test_params(self):
        lin = LinearLayer(n_in=17, n_out=9)
        self.assertIsNotNone(lin.params['W'])
        self.assertIsNotNone(lin.params['b'])

        self.assertIsNotNone(lin.grads['W'])
        self.assertIsNotNone(lin.grads['b'])


if __name__ == "__main__":
    main()