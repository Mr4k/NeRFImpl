import unittest

from nerf import compute_stratified_sample_points

class TestNerf(unittest.TestCase):

    def test_compute_stratified_sample_points_with_two_bins(self):
        for _ in range(1000):
            points = compute_stratified_sample_points(2, 1, 3)
            self.assertEqual(len(points), 2)
            self.assertGreaterEqual(points[0], 1)
            self.assertLess(points[0], 2)
            self.assertGreaterEqual(points[1], 2)
            self.assertLess(points[1], 3)

if __name__ == '__main__':
    unittest.main()