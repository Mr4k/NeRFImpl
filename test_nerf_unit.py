import unittest
import torch

from nerf import compute_stratified_sample_points, trace_ray

class TestNerfUnit(unittest.TestCase):

    def test_compute_stratified_sample_points_with_two_bins(self):
        for _ in range(1000):
            points = compute_stratified_sample_points(2, 1, 3)
            self.assertEqual(len(points), 2)
            self.assertGreaterEqual(points[0], 1)
            self.assertLess(points[0], 2)
            self.assertGreaterEqual(points[1], 2)
            self.assertLess(points[1], 3)

    def test_trace_ray_distance(self):
        for _ in range(1):
            solid_distance = torch.rand(1) * 100

            dir = torch.rand(3)
            dir /= dir.norm()

            #camera_pos = torch.rand(3) * 100
            camera_pos = torch.tensor([0, 0, 0])
            def distance_network(points, dirs):
                colors = []
                opacity = []
                for i in range(points.shape[0]):
                    p = points[i]
                    distance = (p - camera_pos).norm()
                    if distance < solid_distance:
                        opacity.append(0)
                        colors.append(torch.rand(3))
                        continue
                    colors.append(torch.rand(3))
                    opacity.append(10000)
                return colors, opacity
            _, distance = trace_ray(distance_network, camera_pos, dir, 10000, 0.1, 150)
            self.assertLess(distance, solid_distance + 2)
            self.assertGreater(distance, solid_distance - 2)

if __name__ == '__main__':
    unittest.main()