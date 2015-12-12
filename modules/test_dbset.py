import unittest
import numpy as np

from vocab import Vocab
from data_caminfo import DataCamInfo
from modules.dbset import DBSet
from db_dist import DBDist
from nn.utils import check_finite_differences


class TestDBSet(unittest.TestCase):
    def test(self):
        db_data = DataCamInfo()
        db_content = db_data.get_db_for(["area", "food", "pricerange"], "id")

        db_content_name = db_data.get_db_for(["id"], "name")
        db_content_phone = db_data.get_db_for(["id"], "phone")
        db_content_price = db_data.get_db_for(["id"], "pricerange")

        vocab = Vocab()
        entry_vocab = Vocab(no_oov_eos=True)

        for key, val in db_content:
            for key_part in key:
                vocab.add(key_part)
            entry_vocab.add(val)

        for cont in [db_content_phone, db_content_name, db_content_price]:
            for key, val in cont:
                vocab.add(val)

        dbset = DBSet(db_content, [db_content_name, db_content_phone, db_content_price], vocab, entry_vocab)

        input_val = (
            DBDist.get_1hot_from("#west", vocab),
            DBDist.get_1hot_from("#indian", vocab),
            DBDist.get_1hot_from("#expensive", vocab),
        )

        (db_res, aux) = dbset.forward(input_val)
        dinput_val = dbset.backward(aux, tuple(np.random.randn(*x.shape) for x in db_res))

        self.assertEqual(len(dinput_val), len(input_val))
        for di, i in zip(dinput_val, input_val):
            self.assertEqual(di.shape, i.shape)

        self.assertEqual(len(np.where(db_res[0])), len(np.where(db_res[1])))

        for i in np.where(db_res[2])[0]:
            self.assertEqual(vocab.rev(i), "#expensive")

        def gen_input():
            inp = []
            for i in range(3):
                inp.append(np.random.randn(len(vocab)))

            return tuple(inp)

        self.assertTrue(check_finite_differences(
            dbset.forward,
            dbset.backward,
            gen_input_fn=gen_input,
            aux_only=True,
            test_inputs=(0, 1, 2,)
        ))




if __name__ == '__main__':
    unittest.main()
