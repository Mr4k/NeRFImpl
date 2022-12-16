import unittest
import torch

from nerf import compute_stratified_sample_points, generate_ray, trace_ray


class TestNerfUnit(unittest.TestCase):
    def test_compute_stratified_sample_points_with_two_bins(self):
        batch_size = 10
        for _ in range(1000):
            batched_points = compute_stratified_sample_points(batch_size, 2, 1, 3)
            for i in range(batch_size):
                points = batched_points[i]
                self.assertEqual(len(points), 2)
                self.assertGreaterEqual(points[0], 1)
                self.assertLess(points[0], 2)
                self.assertGreaterEqual(points[1], 2)
                self.assertLess(points[1], 3)

    def test_trace_ray_distance(self):
        batch_size = 10
        for _ in range(1000):
            solid_distance = torch.rand(1) * 100

            dirs = torch.rand((batch_size, 3))
            dirs /= dirs.norm(dim=1).reshape(-1, 1).repeat(1, 3)

            camera_pos = torch.rand((batch_size, 3)) * 100

            def distance_network(points, dirs):
                batch_size, num_points, point_dims = points.shape
                colors = []
                opacity = []
                distances = (points - camera_pos.reshape(batch_size, 1, point_dims).repeat(1, num_points, 1)).norm(dim=2)
                opacity = torch.zeros((batch_size, num_points))
                opacity[distances >= solid_distance] = 10000
                colors = torch.zeros((batch_size, num_points, 3))
                return colors, opacity

            _, out_distances = trace_ray(distance_network, camera_pos, dirs, 500, 0.1, 101)
            for d in out_distances:
                self.assertLess(d, solid_distance + 0.5)
                self.assertGreater(d, solid_distance - 0.5)

    def test_trace_ray_distance_far(self):
        dir = torch.tensor([1, 0, 0])

        camera_pos = torch.tensor([2, 2, 3])

        def distance_network(points, dirs):
            colors = []
            opacity = []
            for _ in range(points.shape[0]):
                opacity.append(0)
                colors.append(torch.rand(3))
            return colors, opacity

        _, distance = trace_ray(distance_network, camera_pos, dir, 100, 0.1, 101)
        self.assertAlmostEqual(distance.clone().detach().numpy(), 101, 3)

    def test_trace_ray_distance_near(self):
        dir = torch.tensor([1, 0, 0])

        camera_pos = torch.tensor([2, 2, 3])
        t_near = 30.0

        def distance_network(points, dirs):
            colors = []
            opacity = []
            for _ in range(points.shape[0]):
                opacity.append(10000)
                colors.append(torch.rand(3))
            return colors, opacity

        _, distance = trace_ray(distance_network, camera_pos, dir, 100, t_near, 101)
        self.assertLess(torch.abs(distance - torch.tensor(t_near)), 1.0)

    def test_trace_ray_distance_far(self):
        dir = torch.tensor([1, 0, 0])

        camera_pos = torch.tensor([2, 2, 3])

        def distance_network(points, dirs):
            colors = []
            opacity = []
            for i in range(points.shape[0]):
                opacity.append(0)
                colors.append(torch.rand(3))
            return colors, opacity

        _, distance = trace_ray(distance_network, camera_pos, dir, 100, 0.1, 101)
        self.assertAlmostEqual(distance.clone().detach().numpy(), 101, 3)

    def test_generate_ray(self):
        epsilon = 0.0001
        expected = -torch.tensor([0, 0, 1])
        result = generate_ray(
            torch.tensor(90 * torch.pi / 180), torch.eye(3), 0.5, 0.5, 1
        )
        self.assertLess(
            (expected - result).norm(),
            epsilon,
            f"expected {expected} got {result} instead",
        )

        side_length = torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0))
        expected = -torch.tensor([side_length, 0, side_length])
        result = generate_ray(
            torch.tensor(90 * torch.pi / 180), torch.eye(3), 1, 0.5, 1
        )
        self.assertLess(
            (expected - result).norm(),
            epsilon,
            f"expected {expected} got {result} instead",
        )

        side_length = torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0))
        expected = -torch.tensor([0, side_length, side_length])
        result = generate_ray(
            torch.tensor(90 * torch.pi / 180), torch.eye(3), 0.5, 1, 1
        )
        self.assertLess(
            (expected - result).norm(),
            epsilon,
            f"expected {expected} got {result} instead",
        )

        expected = -torch.tensor(
            [
                torch.sin(torch.tensor(90 * torch.pi / 180) / 4.0),
                0,
                torch.cos(torch.tensor(90 * torch.pi / 180) / 4.0),
            ]
        )
        result = generate_ray(
            torch.tensor(45 * torch.pi / 180), torch.eye(3), 1.0, 0.5, 1
        )
        self.assertLess(
            (expected - result).norm(),
            epsilon,
            f"expected {expected} got {result} instead",
        )

        # test invariant to scale
        side_length = torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0))
        expected = -torch.tensor([0, side_length, side_length])
        result = generate_ray(
            torch.tensor(90 * torch.pi / 180), torch.eye(3) * 4.0, 0.5, 1, 1
        )
        self.assertLess(
            (expected - result).norm(),
            epsilon,
            f"expected {expected} got {result} instead",
        )


if __name__ == "__main__":
    unittest.main()
