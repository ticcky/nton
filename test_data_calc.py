import unittest

from data_calc import DataCalc


class TestDataCalc(unittest.TestCase):
    def test_data(self):
        d = DataCalc()
        train_data = d.gen_data()
        test_data = d.gen_data(test_data=True)

        print 'train data'
        for i in range(10):
            print next(train_data)

        print 'test data'
        for i in range(10):
            print next(test_data)



if __name__ == '__main__':
    unittest.main()
