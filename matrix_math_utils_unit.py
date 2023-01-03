import unittest
import torch

from matrix_math_utils import generate_random_hemisphere_gimbal_transformation_matrix
from nerf import generate_rays, get_camera_position


class TestMatrixMathUtils(unittest.TestCase):
    def test_generate_random_hemisphere_gimbal_transformation_matrix(self):
        scale = 5

        for _ in range(1000):
            transformation_matrix = generate_random_hemisphere_gimbal_transformation_matrix(scale)
            point = get_camera_position(transformation_matrix)
            center_ray = generate_rays(
                torch.tensor(torch.pi * 45 / 180),
                transformation_matrix[:3, :3],
                torch.tensor([[0.5, 0.5]]),
            )

            self.assertAlmostEqual(point.norm().item(), scale, 3)
            self.assertAlmostEqual((point + center_ray * scale).norm().item(), 0, 3)


if __name__ == "__main__":
    unittest.main()
