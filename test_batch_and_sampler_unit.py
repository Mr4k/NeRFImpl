import torch

import unittest

from batch_and_sampler import sample_batch

class TestBatchAndSamplerUnit(unittest.TestCase):
    def test_sample_batch(self):
        matricies = [
            torch.rand([4, 4]),
            torch.rand([4, 4]),
            torch.rand([4, 4]),
            torch.rand([4, 4]),
            torch.rand([4, 4]),
            torch.rand([4, 4]),
            torch.rand([4, 4]),
        ]
        camera_poses, rays, distance_to_depth_modifiers = sample_batch(4096, 200, matricies, torch.tensor(45 * torch.pi / 180))
        self.assertEqual(camera_poses.shape, torch.Size([4096, 3]))
        self.assertEqual(rays.shape, torch.Size([4096, 3]))
        self.assertEqual(distance_to_depth_modifiers.shape, torch.Size([4096]))

if __name__ == "__main__":
    unittest.main()
