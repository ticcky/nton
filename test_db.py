from unittest import TestCase, main
import numpy as np

from db import DB
from nn.utils import check_finite_differences

class TestDB(TestCase):
    def test_forward(self):
        db = DB()
        #db.backward(np.array([0.7, 0.1, 0.1, 0.1]), np.array([1.0, 1.0, 1.0, 1.0]))
        czech = db.get_vector('czech')

        ((db_y, ), aux) = db.forward((czech, ))

        (dczech, ) = db.backward((czech, ), aux, (np.random.randn(*db_y.shape), ))


    def test_backward(self):
        db = DB()

        def gen_input():
            food = np.random.choice(['chinese', 'czech', 'english', 'indian'])
            x = db.get_vector(food)

            return (x, )

        check = check_finite_differences(
            db.forward,
            db.backward,
            gen_input_fn=gen_input
        )
        self.assertTrue(check)

        #print db.forward(['czech'], [1.0])
        #print db.forward(['indian'], [1.0])
        #print db.forward(['czech', 'indian'], [0.6, 0.4])
        #print db.forward(['czech', 'indian', 'chinese'], [0.1, 0.4, 0.5])
        #db.forward('indian')


if __name__ == "__main__":
    main()