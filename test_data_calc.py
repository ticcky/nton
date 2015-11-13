import unittest

from data_calc import DataCalc


class TestDataCalc(unittest.TestCase):
    def test_data(self):
        d = DataCalc()
        train_data = d.gen_data()
        test_data = d.gen_data(test_data=True)

        print '### train data'
        for i in range(10):
            q, a = next(train_data)
            print "Q:", " ".join(q)
            print "A:", " ".join(a)
            print

        print '### test data'
        for i in range(10):
            q, a = next(test_data)
            print "Q:", " ".join(q)
            print "A:", " ".join(a)
            print



if __name__ == '__main__':
    unittest.main()
