import unittest
import numpy as np

from db_dist import DBDist
from data_caminfo import DataCamInfo
from vocab import Vocab

from nn.utils import check_finite_differences


class TestDBDist(unittest.TestCase):
    def test_forward(self):
        db_data = DataCamInfo()
        db_content = db_data.get_db_for(["area", "food", "pricerange"], "id")

        vocab = Vocab()
        for key, val in db_content:
            for key_part in key:
                vocab.add(key_part)

        dbdist = DBDist(db_content, vocab)

        val1 = dbdist.get_vector("#west")
        val2 = dbdist.get_vector("#indian")
        val3 = dbdist.get_vector("#expensive")

        ((db_res, ), aux) = dbdist.forward((val1, val2, val3))

        for i in np.where(db_res)[0]:
            key, ndx = db_content[i]
            self.assertEqual(key, ("#west", "#indian", "#expensive"))

        def gen_input():
            inp = []
            for i in range(3):
                inp.append(np.random.randn(len(vocab)))

            return tuple(inp)

        check = check_finite_differences(
            dbdist.forward,
            dbdist.backward,
            gen_input_fn=gen_input,
            aux_only=True
        )

        self.assertTrue(check)




if __name__ == '__main__':
    unittest.main()
