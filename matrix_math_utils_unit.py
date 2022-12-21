import unittest
import torch

from matrix_math_utils import generate_random_gimbal_transformation_matrix
from nerf import get_camera_position

class TestMatrixMathUtils(unittest.TestCase):
    def test_generate_random_gimbal_transformation(self):
        scale = 5

        for _ in range(1000):
            transformation_matrix = generate_random_gimbal_transformation_matrix(scale)
            point = get_camera_position(transformation_matrix)

            self.assertAlmostEquals(point.norm().item(), scale, 3)

if __name__ == "__main__":
    unittest.main()
