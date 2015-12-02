import unittest
import numpy as np
import random

from data_caminfo import DataCamInfo
from dbn import DBN
from nn.utils import check_finite_differences

class TestDBN(unittest.TestCase):
    def test(self):
        db_data = DataCamInfo()
        db_content = db_data.get_db_for(["area", "food", "pricerange"], "name")

        vocab = db_data.get_vocab()

        db = DBN(db_content, vocab)

        #any = db.get_vector('0') * 0 + 1

        for entry in db_data.db_content:
            empty_result = False
            for i in range(5):
                db_input = (
                    db.get_vector(db_data.get_tagged_value(entry['area'])),
                    db.get_vector(db_data.get_tagged_value(entry['food'])),
                    db.get_vector(db_data.get_tagged_value(entry['pricerange'])),
                )
                ((db_res, ), _) = db.forward(db_input)

                matching = set(db_data.get_tagged_value(e['name']) for e in db_data.db_content if e['area'] == entry['area'] and e['food'] == entry['food'] and e['pricerange'] == entry['pricerange'] and e)
                returned = list(db.vocab.rev(i) for i in np.where(db_res == 1)[0])

                if returned:
                    self.assertTrue(empty_result == False)
                    self.assertTrue(returned[0] in matching)
                else:
                    empty_result = True

            self.assertFalse(empty_result)

        def gen_input():
            res = []
            v = db.get_vector()
            for i in range(db.n):
                res.append(np.random.randn(*v.shape))

            return tuple(res)

        check = check_finite_differences(
            db.forward,
            db.backward,
            gen_input_fn=gen_input,
            aux_only=True
        )



        self.assertTrue(check)


if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
