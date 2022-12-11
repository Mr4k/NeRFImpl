import unittest
import torch

from nerf import trace_ray  

class TestNerfUnit(unittest.TestCase):

    def cube_network(network, points, dirs):
        results = []
        side_length = 1
        half_length = side_length / 2.0
        large = 10000
        for p in points:
            x, y, z = p
            if torch.abs(x) < half_length and torch.abs(y) < half_length or torch.abs(z) < half_length:
                 results.append((large, torch.tensor([0, 0, 0])))
                 continue
            results.append((0, torch.tensor([0, 0, 0])))
        return results
       

    def test_rendering_depth_e2e_with_given_network(self):
        pass
            

if __name__ == '__main__':
    unittest.main()