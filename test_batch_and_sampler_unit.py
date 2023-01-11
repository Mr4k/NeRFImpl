import torch

import unittest

from batch_and_sampler import random_partition, sample_batch


class TestBatchAndSamplerUnit(unittest.TestCase):
    def test_sample_batch(self):
        frame_len = 7
        matricies = torch.stack(
            [
                torch.rand([4, 4]),
                torch.rand([4, 4]),
                torch.rand([4, 4]),
                torch.rand([4, 4]),
                torch.rand([4, 4]),
                torch.rand([4, 4]),
                torch.rand([4, 4]),
            ]
        )
        batch_size = 4096
        size = 200

        camera_poses, rays, distance_to_depth_modifiers, expected_colors = sample_batch(
            batch_size,
            size,
            matricies,
            torch.rand([frame_len, size, size, 3]),
            torch.tensor(45 * torch.pi / 180),
        )
        self.assertEqual(camera_poses.shape, torch.Size([batch_size, 3]))
        self.assertEqual(rays.shape, torch.Size([batch_size, 3]))
        self.assertEqual(distance_to_depth_modifiers.shape, torch.Size([batch_size]))
        self.assertEqual(expected_colors.shape, torch.Size([batch_size, 3]))

    def test_random_partition_more_draws_than_items(self):
        for _ in range(1000):
            num_draws = torch.randint(10, 1000, [1]).item()
            num_items = torch.randint(1, num_draws, [1]).item()
            partition = random_partition(num_items, num_draws)
            self.assertEqual(partition.shape, torch.Size([num_items]))
            self.assertLessEqual(torch.max(partition), num_draws)
            self.assertGreaterEqual(torch.min(partition), 0)


if __name__ == "__main__":
    unittest.main()
