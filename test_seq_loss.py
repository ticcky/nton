from unittest import TestCase, main
import numpy as np

from nn.utils import check_finite_differences

from seq_loss import SeqLoss


class TestSeqLoss(TestCase):
    def test_forward(self):
        y_hat = np.array(
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [0.3, 0.7]
            ]
        )
        y_true = np.array([1, 0, 0])

        ((res, ), aux, ) = SeqLoss.forward((y_hat, y_true))

        self.assertTrue(np.allclose(res, (- np.log(1.0) - np.log(0.5) - np.log(0.3)) / 3.0))


    def test_backward(self):
        self.assertTrue(
            check_finite_differences(
                SeqLoss.forward,
                SeqLoss.backward,
                gen_input_fn=lambda: (np.random.dirichlet([1, 1], (10, )), [np.random.binomial(1, 0.5) for i in range(10)] ),
                aux_only=True
            )
        )




if __name__ == '__main__':
    main()