import unittest

from metrics import calculate_wer

class TestMetrics(unittest.TestCase):
    def test_wer(self):
        self.assertEqual(calculate_wer([0, 1, 2], [0, 1, 2]), 0)
        self.assertEqual(calculate_wer([0, 1, 2], [0, 1]), 1.0 / 3)
        self.assertEqual(calculate_wer([0, 1, 2], [1, 2, 0]), 2.0 / 3)


if __name__ == '__main__':
    unittest.main()
