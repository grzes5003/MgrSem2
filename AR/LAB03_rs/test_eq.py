import sys
import unittest
import numpy.testing as nt
import numpy as np


# "target/debug/results/result.npy" "results/result.npy"
class TestStringMethods(unittest.TestCase):
    def test_arrays(self):
        # actual_path = sys.argv[1]
        # expected_path = sys.argv[2]
        actual_path = "target/debug/results/result.npy"
        expected_path = "results/result.npy"

        actual_arr = np.load(actual_path)
        expected_arr = np.load(expected_path)

        self.assertIsNone(nt.assert_equal(actual_arr, expected_arr))


if __name__ == '__main__':
    unittest.main()

