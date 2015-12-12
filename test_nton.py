import unittest
import numpy as np

from data_calc import DataCalc
from db import DB
from nn import OneHot
from nton import NTON
from nn.utils import check_finite_differences, TestParamGradInLayer
from vocab import Vocab
from data_caminfo import DataCamInfo


class TestNTON(unittest.TestCase):
    def test(self):
        db_data = DataCamInfo()

        db_content = db_data.get_db_for(["area", "food", "pricerange"], "id")

        db_content_name = db_data.get_db_for(["id"], "name")
        db_content_phone = db_data.get_db_for(["id"], "phone")
        db_content_price = db_data.get_db_for(["id"], "pricerange")

        vocab = Vocab()
        vocab.add('<start>')
        db_mapping = [
            vocab.add("<slu_area>"),
            vocab.add("<slu_food>"),
            vocab.add("<slu_pricerange>"),
            vocab.add("<tr_area>"),
            vocab.add("<tr_food>"),
            vocab.add("<tr_pricerange>"),
            vocab.add("<db_area>"),
            vocab.add("<db_food>"),
            vocab.add("<db_pricerange>"),
        ]
        entry_vocab = Vocab(no_oov_eos=True)

        for key, val in db_content:
            for key_part in key:
                vocab.add(key_part)
            entry_vocab.add(val)

        for cont in [db_content_phone, db_content_name, db_content_price]:
            for key, val in cont:
                vocab.add(val)

        dialogs = []
        for _, dialog in zip(range(5), db_data.gen_data()):
            dialogs.append(dialog)

            for sys, usr in dialog:
                for word in sys.split() + usr.split():
                    vocab.add(word)

        nton = NTON(5, 5, db_content, [db_content_name, db_content_phone, db_content_price], db_mapping, vocab, entry_vocab)
        for dialog in dialogs:
            print 'Dialog 1'
            nton.forward(dialog)



if __name__ == '__main__':
    import random
    import util
    util.pdb_on_error()
    random.seed(0)
    np.random.seed(0)

    unittest.main()

