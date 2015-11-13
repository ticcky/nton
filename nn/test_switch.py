import numpy as np
from unittest import TestCase, main

from nn.switch import Switch
from nn.utils import TestParamGradInLayer, check_finite_differences
from nn.vars import Vars


class TestSwitch(TestCase):
    def test_forward(self):
        p1 = np.random.random()
        in1 = np.random.randn(13)
        in2 = np.random.randn(13)

        ((out, ), aux) = Switch.forward((p1, in1, in2))
        self.assertEqual(out.shape, (13, ))

    def test_backward(self):
        def gen_input():
            p1 = np.random.randn(1)
            in1 = np.random.randn(13)
            in2 = np.random.randn(13)

            return (p1, in1, in2, )

        check = check_finite_differences(
            Switch.forward,
            Switch.backward,
            gen_input_fn=gen_input,
            test_inputs=(0, 1, 2),
            aux_only=True
        )
        self.assertTrue(check)


if __name__ == "__main__":
    main()