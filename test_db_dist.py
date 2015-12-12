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

        db_content_name = db_data.get_db_for(["id"], "name")
        db_content_phone = db_data.get_db_for(["id"], "phone")

        vocab = Vocab()
        entry_vocab = Vocab(no_oov_eos=True)

        for key, val in db_content:
            for key_part in key:
                vocab.add(key_part)
            entry_vocab.add(val)

        for cont in [db_content_phone, db_content_name]:
            for key, val in cont:
                vocab.add(val)

        dbdist = DBDist(db_content, vocab, entry_vocab)
        dbdist_name = DBDist(db_content_name, entry_vocab, vocab)
        dbdist_phone = DBDist(db_content_phone, entry_vocab, vocab)

        val1 = dbdist.get_vector("#west")
        val2 = dbdist.get_vector("#indian")
        val3 = dbdist.get_vector("#expensive")

        ((db_res, ), aux) = dbdist.forward((val1, val2, val3))
        ((db_res_name, ), aux) = dbdist_name.forward((db_res, ))
        ((db_res_phone, ), aux) = dbdist_phone.forward((db_res, ))

        self.assertTrue(len(np.where(db_res_name)) == len(np.where(db_res_phone)))
        self.assertTrue(len(np.where(db_res)) == len(np.where(db_res_phone)))

        for i in np.where(db_res)[0]:
            key, ndx = db_content[entry_vocab[i]]
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
