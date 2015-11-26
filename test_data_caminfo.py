import unittest

from data_caminfo import DataCamInfo


class TestDataCaminfo(unittest.TestCase):
    def test(self):
        db = DataCamInfo()
        db_content = db.get_db()
        for entry in db_content:
            self.assertTrue(len(entry) == len(db.fields))

        vocab = db.get_vocab()
        data = db.gen_data()
        for i in range(10):
            q, a = next(data)
            self.assertTrue(q != '')
            self.assertTrue(a != '')



if __name__ == '__main__':
    unittest.main()
