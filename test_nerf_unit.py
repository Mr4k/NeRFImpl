import unittest
import torch

from nerf import compute_stratified_sample_points, generate_ray, trace_ray

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
        for _ in range(1000):
            solid_distance = torch.rand(1) * 100

            dir = torch.rand(3)
            dir /= dir.norm()

            camera_pos = torch.rand(3) * 100

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
            _, distance = trace_ray(distance_network, camera_pos, dir, 500, 0.1, 101)
            self.assertLess(distance, solid_distance + 0.5)
            self.assertGreater(distance, solid_distance - 0.5)

    def test_generate_ray(self):
        epsilon = 0.0001
        expected = torch.tensor([0, 0, 1])
        result = generate_ray(torch.tensor(90 * torch.pi / 180), torch.eye(3), 0.5, 0.5, 1)
        self.assertLess((expected - result).norm(), epsilon, f"expected {expected} got {result} instead")

        side_length = torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0))
        expected = torch.tensor([side_length, 0, side_length])
        result = generate_ray(torch.tensor(90 * torch.pi / 180), torch.eye(3), 1, 0.5, 1)
        self.assertLess((expected - result).norm(), epsilon, f"expected {expected} got {result} instead")

        side_length = torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0))
        expected = torch.tensor([0, side_length, side_length])
        result = generate_ray(torch.tensor(90 * torch.pi / 180), torch.eye(3), 0.5, 1, 1)
        self.assertLess((expected - result).norm(), epsilon, f"expected {expected} got {result} instead")

        expected = torch.tensor(
            [
                torch.sin(torch.tensor(90 * torch.pi / 180) / 4.0),
                0,
                torch.cos(torch.tensor(90 * torch.pi / 180) / 4.0)
            ]
        )
        result = generate_ray(torch.tensor(45 * torch.pi / 180), torch.eye(3), 1.0, 0.5, 1)
        self.assertLess((expected - result).norm(), epsilon, f"expected {expected} got {result} instead")

        # test invariant to scale
        side_length = torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0))
        expected = torch.tensor([0, side_length, side_length])
        result = generate_ray(torch.tensor(90 * torch.pi / 180), torch.eye(3) * 4.0, 0.5, 1, 1)
        self.assertLess((expected - result).norm(), epsilon, f"expected {expected} got {result} instead")
    


if __name__ == '__main__':
    unittest.main()