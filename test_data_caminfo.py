import unittest
import numpy as np
import random

from data_caminfo import DataCamInfo


class TestDataCaminfo(unittest.TestCase):
    def test(self):
        db = DataCamInfo()
        db_content = db.get_db()
        for entry in db_content:
            self.assertTrue(len(entry) == len(db.fields))

        data = db.get_db_for(['food', 'area', 'pricerange'], 'name')

        vocab = db.get_vocab()
        data = db.gen_data()
        for i in range(10):
            q, a = next(data)
            self.assertTrue(q != '')
            self.assertTrue(a != '')

            print " ".join(q)
            print " ".join(a)
            print



if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    unittest.main()
