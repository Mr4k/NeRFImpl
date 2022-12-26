import unittest
import torch

from nerf import (
    compute_stratified_sample_points,
    generate_rays,
    inverse_transform_sampling,
    trace_hierarchical_ray,
    trace_ray,
)


class TestNerfUnit(unittest.TestCase):
    def test_compute_stratified_sample_points_with_two_bins(self):
        batch_size = 10
        for _ in range(1000):
            batched_points = compute_stratified_sample_points(
                batch_size,
                2,
                torch.tensor([1]).repeat(batch_size),
                torch.tensor([3]).repeat(batch_size),
            )
            for i in range(batch_size):
                points = batched_points[i]
                self.assertEqual(len(points), 2)
                self.assertGreaterEqual(points[0], 1)
                self.assertLess(points[0], 2)
                self.assertGreaterEqual(points[1], 2)
                self.assertLess(points[1], 3)

    def test_trace_ray_distance(self):
        batch_size = 10
        for _ in range(100):
            solid_distance = torch.rand(1) * 100

            dirs = torch.rand((batch_size, 3))
            dirs /= dirs.norm(dim=1).reshape(-1, 1).repeat(1, 3)

            num_samples = 500

            camera_pos = torch.rand((batch_size, 3)) * 100

            class DistanceNetwork:
                def to(self, _):
                    return self

                def __call__(self, points, dirs):
                    num_points, _ = points.shape
                    distances = (
                        points - camera_pos.repeat_interleave(num_samples + 1, 0)
                    ).norm(dim=1)
                    opacity = torch.zeros((num_points))
                    opacity[distances >= solid_distance] = 10000
                    colors = torch.zeros((num_points, 3))
                    return colors, opacity

            _, out_distances, _ = trace_ray(
                "cpu",
                DistanceNetwork(),
                camera_pos,
                dirs,
                num_samples,
                torch.tensor([0.1]).repeat(batch_size),
                torch.tensor([101]).repeat(batch_size),
            )
            for d in out_distances:
                self.assertLess(d, solid_distance + 0.5)
                self.assertGreater(d, solid_distance - 0.5)

    def test_trace_ray_distance_far(self):
        dirs = torch.tensor([[1, 0, 0]])

        camera_poses = torch.tensor([[2, 2, 3]])

        class DistanceNetwork:
            def to(self, _):
                return self

            def __call__(self, points, dirs):
                batch_size, _ = points.shape

                opacity = torch.zeros((batch_size))
                colors = torch.zeros((batch_size, 3))
                return colors, opacity

        _, distance, _ = trace_ray(
            "cpu",
            DistanceNetwork(),
            camera_poses,
            dirs,
            100,
            torch.tensor([0.1]),
            torch.tensor([101]),
        )
        self.assertAlmostEqual(distance.clone().detach().numpy()[0], 101, 3)

    def test_trace_ray_distance_near(self):
        dirs = torch.tensor([[1, 0, 0]])

        camera_poses = torch.tensor([[2, 2, 3]])
        t_near = 30.0

        class DistanceNetwork:
            def to(self, _):
                return self

            def __call__(self, points, dirs):
                batch_size, _ = points.shape

                opacity = torch.ones((batch_size)) * 10000
                colors = torch.zeros((batch_size, 3))
                return colors, opacity

        _, distance, _ = trace_ray(
            "cpu",
            DistanceNetwork(),
            camera_poses,
            dirs,
            100,
            torch.tensor([t_near]),
            torch.tensor([101]),
        )
        self.assertLess(torch.abs(distance[0] - torch.tensor(t_near)), 1.0)

    def test_trace_hierarchical_ray_distance(self):
        batch_size = 10
        for _ in range(100):
            solid_distance = torch.rand(1) * 100

            dirs = torch.rand((batch_size, 3))
            dirs /= dirs.norm(dim=1).reshape(-1, 1).repeat(1, 3)

            num_coarse_samples = 400
            num_fine_samples = 50

            camera_pos = torch.rand((batch_size, 3)) * 100

            class FineDistanceNetwork:
                def to(self, _):
                    return self

                def __call__(self, points, dirs):
                    num_points, _ = points.shape
                    distances = (
                        points - camera_pos.repeat_interleave(num_coarse_samples + num_fine_samples + 1, 0)
                    ).norm(dim=1)
                    opacity = torch.zeros((num_points))
                    opacity[distances >= solid_distance] = 10000
                    colors = torch.zeros((num_points, 3))
                    return colors, opacity

            class CoarseDistanceNetwork:
                def to(self, _):
                    return self

                def __call__(self, points, dirs):
                    num_points, _ = points.shape
                    distances = (
                        points - camera_pos.repeat_interleave(num_coarse_samples + 1, 0)
                    ).norm(dim=1)
                    opacity = torch.zeros((num_points))
                    # intentional start allocating some probability at solid_distance / 2 so that we can be sure the fine
                    # network follows through
                    opacity[distances >= solid_distance / 2] = 10
                    colors = torch.zeros((num_points, 3))
                    return colors, opacity

            _, out_distances, _ = trace_hierarchical_ray(
                "cpu",
                CoarseDistanceNetwork(),
                FineDistanceNetwork(),
                camera_pos,
                dirs,
                num_coarse_samples,
                num_fine_samples,
                torch.tensor([0.1]).repeat(batch_size),
                torch.tensor([101]).repeat(batch_size),
            )
            for d in out_distances:
                self.assertLess(d, solid_distance + 0.5)
                self.assertGreater(d, solid_distance - 0.5)

    def test_generate_rays_multibatch(self):
        epsilon = 0.0001
        side_length = torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0))
        batch_size = 4
        expected = torch.stack(
            [
                -torch.tensor([0, 0, 1]),
                -torch.tensor([side_length, 0, side_length]),
                -torch.tensor([0, side_length, side_length]),
                -torch.tensor([0, 0, 1]),
            ]
        )

        result = generate_rays(
            torch.tensor(90 * torch.pi / 180),
            torch.eye(3),
            torch.tensor([[0.5, 0.5], [1.0, 0.5], [0.5, 1.0], [0.5, 0.5]]),
            1,
        )
        self.assertEqual(result.shape, torch.Size([4, 3]))
        self.assertLess(
            (expected - result).norm(dim=1).sum() / batch_size,
            epsilon,
            f"expected {expected} got {result} instead",
        )

    def test_generate_rays(self):
        epsilon = 0.0001
        expected = -torch.tensor([0, 0, 1])
        result = generate_rays(
            torch.tensor(90 * torch.pi / 180),
            torch.eye(3),
            torch.tensor([[0.5, 0.5]]),
            1,
        )
        self.assertLess(
            (expected - result).norm(),
            epsilon,
            f"expected {expected} got {result} instead",
        )

        side_length = torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0))
        expected = -torch.tensor([side_length, 0, side_length])
        result = generate_rays(
            torch.tensor(90 * torch.pi / 180), torch.eye(3), torch.tensor([[1, 0.5]]), 1
        )
        self.assertLess(
            (expected - result).norm(),
            epsilon,
            f"expected {expected} got {result} instead",
        )

        side_length = torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0))
        expected = -torch.tensor([0, side_length, side_length])
        result = generate_rays(
            torch.tensor(90 * torch.pi / 180), torch.eye(3), torch.tensor([[0.5, 1]]), 1
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
        result = generate_rays(
            torch.tensor(45 * torch.pi / 180),
            torch.eye(3),
            torch.tensor([[1.0, 0.5]]),
            1,
        )
        self.assertLess(
            (expected - result).norm(),
            epsilon,
            f"expected {expected} got {result} instead",
        )

        # test invariant to scale
        side_length = torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0))
        expected = -torch.tensor([0, side_length, side_length])
        result = generate_rays(
            torch.tensor(90 * torch.pi / 180),
            torch.eye(3) * 4.0,
            torch.tensor([[0.5, 1]]),
            1,
        )
        self.assertLess(
            (expected - result).norm(),
            epsilon,
            f"expected {expected} got {result} instead",
        )

    def test_inverse_transform_sampling(self):
        batch_size = 1
        stopping_probs = torch.tensor(
            [
                [0.05, 0.5, 0.3, 0.15],
                [0.3, 0.4, 0.1, 0.2],
            ]
        )
        bin_boundaries = torch.tensor(
            [
                [0.0, 1, 2, 3, 4],
                [0.0, 10, 20, 70, 100],
            ]
        )
        points = inverse_transform_sampling(stopping_probs, bin_boundaries, 1000)

        for i in range(batch_size):
            bin_counts, _ = torch.histogram(points[i], bin_boundaries[i])
            self.assertLess(
                (bin_counts / bin_counts.sum() - stopping_probs[i]).abs().sum(), 0.05
            )


if __name__ == "__main__":
    unittest.main()
