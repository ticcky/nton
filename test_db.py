from unittest import TestCase, main
import numpy as np

from db import DB

class TestDB(TestCase):
    def test_forward(self):
        db = DB()
        #db.backward(np.array([0.7, 0.1, 0.1, 0.1]), np.array([1.0, 1.0, 1.0, 1.0]))
        czech = db.get_vector('czech')
        db.forward(czech)
        db.backward(czech)
        print db.forward(['czech'], [1.0])
        print db.forward(['indian'], [1.0])
        print db.forward(['czech', 'indian'], [0.6, 0.4])
        print db.forward(['czech', 'indian', 'chinese'], [0.1, 0.4, 0.5])
        #db.forward('indian')


if __name__ == "__main__":
    main()