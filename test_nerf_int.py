import unittest
import torch

import imageio.v3 as iio
import imageio

import os

from nerf import generate_ray, get_camera_position, load_config_file, trace_ray  

from tqdm import tqdm

class TestNerfUnit(unittest.TestCase):

    def test_rendering_depth_e2e_with_given_network(self):
        def cube_network(points, dirs):
            colors = []
            opacity = []
            for i in range(points.shape[0]):
                p = points[i]
                if torch.max(torch.abs(p)) <= 1.0:
                    opacity.append(10000)
                    colors.append(torch.rand(3))
                    continue
                colors.append(torch.rand(3))
                opacity.append(0)
            return colors, opacity


        frames = load_config_file("./integration_test_data/cameras.json")
        near = 0.5
        far = 7
        for f in frames:
            fov = torch.tensor(f["fov"])
            transformation_matrix = torch.tensor(f["transformation_matrix"], dtype=torch.float).t()
            image_src = f["file_path"]

            camera_pos = get_camera_position(transformation_matrix)
            
            fname, _ = image_src.split(".png")
            ext = "png"
            image_src = fname + "_depth" + "." + ext

            pixels = torch.tensor(iio.imread(os.path.join("./integration_test_data/", image_src))).fliplr().float()
            width, height = pixels.shape

            center_ray = generate_ray(fov, transformation_matrix[:3, :3], 0.5, 0.5, 1)

            size = 100
            result = torch.zeros((size, size), dtype=torch.int)
            total_depth_error = 0
            for x in tqdm(range(size)):
                for y in range(size):
                    s_x = x / size + 0.5 / size
                    s_y = y / size + 0.5 / size
                    ray = generate_ray(fov, transformation_matrix[:3, :3], s_x, s_y, 1)
                    distance_to_depth_modifier = torch.dot(ray, center_ray)
                    
                    expected_depth = (1 - pixels[int(height * s_y), int(width * s_x)] / 255.0) * (far - near) + near
                    _, dist = trace_ray(cube_network, camera_pos, ray, 100, near / distance_to_depth_modifier, far / distance_to_depth_modifier)
                    depth = dist * distance_to_depth_modifier
                    normalized_depth = (depth - near) / (far - near)
                    inverted_normalized_depth = 1 - normalized_depth
                    result[x, y] = int(inverted_normalized_depth * 255)
                    total_depth_error += torch.abs(depth - expected_depth)

            average_depth_error = total_depth_error / size / size
            expected_average_depth_error = 0.05
            self.assertLess(average_depth_error, expected_average_depth_error, f"expected avg depth error {average_depth_error} to be smaller than {expected_average_depth_error}")
            imageio.imwrite("recon" + image_src, result.t().fliplr().numpy())
        pass
            

if __name__ == '__main__':
    unittest.main()