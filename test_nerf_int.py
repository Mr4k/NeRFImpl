import unittest
import torch

import imageio.v3 as iio

import os

from nerf import generate_ray, get_camera_position, load_config_file, trace_ray  

class TestNerfUnit(unittest.TestCase):

    def test_rendering_depth_e2e_with_given_network(self):
        def cube_network(points, dirs):
            colors = []
            opacity = []
            for _ in range(points.shape[0]):
                if torch.max(points) < 1.0:
                    opacity.append(10000)
                    colors.append(torch.rand(3))
                    continue
                colors.append(torch.rand(3))
                opacity.append(0)
            return colors, opacity


        frames = load_config_file("./integration_test_data/cameras.json")
        samples_per_frame = 100
        near = 0.1
        far = 10
        for f in frames:
            fov = torch.tensor(f["fov"])
            transformation_matrix = torch.tensor(f["transformation_matrix"], dtype=torch.float)
            image_src = f["file_path"]

            camera_pos = get_camera_position(transformation_matrix)
            print("camera pos:", camera_pos)
            
            pixels = iio.imread(os.path.join("./integration_test_data/", image_src))
            width, height, _ = pixels.shape

            for _ in range(samples_per_frame):
                s_x, s_y = torch.rand(2)
                ray = generate_ray(fov, transformation_matrix[:3, :3], s_x, s_y, 1)
                print(ray.shape)
                distance_to_depth_modifier = torch.dot(ray, ray)
                
                expected_depth = pixels[int(width * s_x), int(height * s_y), 0] / 255.0
                _, dist = trace_ray(cube_network, camera_pos, ray, 100, near, far)
                depth = dist / distance_to_depth_modifier / (far - near)
                self.assertLess(torch.abs(depth - expected_depth), 0.01, f"expected depth {expected_depth} actual {depth}")
        pass
            

if __name__ == '__main__':
    unittest.main()