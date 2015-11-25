from unittest import TestCase, main
import numpy as np
import random

from db2 import DB2
from vocab import Vocab
from nn.utils import check_finite_differences


class TestDB2(TestCase):
    def test_forward_backward(self):
        vocab = Vocab()
        for i in range(19):
            vocab.add(str(i))
            #print str(i), '->', vocab.add(str(i)), vocab[str(i)]

        content = []
        for i in range(10):
            for y in range(10):
                content.append((str(i), str(y), str(i + y)))

        db = DB2(content, vocab)
        for e1, r, r2 in [(1, 1, 2), (1, 2, 3), (1, 5, 6), (5, 1, 6)]:
            ((y, ), aux) = db.forward((db.get_vector(str(e1)), db.get_vector(str(r))))

            dy = np.random.randn(*y.shape)
            (de1, dr) = db.backward(aux, (dy, ))

        def gen_input():
            (e1, r, e2) = random.choice(db.content)
            v_e1 = db.get_vector(str(e1))
            v_e1 += np.random.randn(*v_e1.shape)
            v_r = db.get_vector(str(r))
            v_r += np.random.randn(*v_r.shape)

            return (v_e1, v_r, )

        check = check_finite_differences(
            db.forward,
            db.backward,
            gen_input_fn=gen_input,
            aux_only=True
        )

        self.assertTrue(check)


if __name__ == '__main__':
    main()
