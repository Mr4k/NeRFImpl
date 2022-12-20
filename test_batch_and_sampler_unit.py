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
        result = sample_batch(4096, 200, matricies, 45 * torch.pi / 180, 0.1, 10)
        self.assertEqual(result.shape, torch.Size([4096, 3]))

if __name__ == "__main__":
    unittest.main()
