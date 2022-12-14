import unittest
import torch

import imageio.v3 as iio
import imageio

import os

from nerf import generate_ray, get_camera_position, load_config_file, trace_ray  

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
        samples_per_frame = 100
        near = 0.5
        far = 7
        for f in frames:
            fov = torch.tensor(f["fov"])
            transformation_matrix = torch.tensor(f["transformation_matrix"], dtype=torch.float).t()
            image_src = f["file_path"]

            camera_pos = get_camera_position(transformation_matrix)
            print("camera pos:", camera_pos, camera_pos.norm())

            print("t matrix:", transformation_matrix)
            
            print(image_src.split(".png"))
            fname, _ = image_src.split(".png")
            ext = "png"
            image_src = fname + "_depth" + "." + ext

            print("path:", image_src)
            pixels = iio.imread(os.path.join("./integration_test_data/", image_src))
            width, height = pixels.shape

            center_ray = generate_ray(fov, transformation_matrix[:3, :3], 0.5, 0.5, 1)

            size = 100
            result = torch.zeros((size, size), dtype=torch.int)
            for x in range(size):
                for y in range(size):
                    print("x:", x, "y:", y)
                    #for _ in range(samples_per_frame):
                    #s_x, s_y = torch.rand(2)
                    #s_x, s_y = 0.5, 0.5
                    s_x = x / size + 0.5 / size
                    s_y = y / size + 0.5 / size
                    ray = generate_ray(fov, transformation_matrix[:3, :3], s_x, s_y, 1)
                    print("ray:", ray)
                    distance_to_depth_modifier = torch.dot(ray, center_ray)
                    
                    expected_depth = pixels[int(width * s_x), int(height * s_y)] / 255.0
                    _, dist = trace_ray(cube_network, camera_pos, ray, 100, near, far)
                    print(dist, distance_to_depth_modifier)
                    depth = 1 - ((dist * distance_to_depth_modifier) - near) / (far - near)
                    print("d:", depth, s_x, s_y)
                    if (int(depth * 255) < 254):
                        result[x, y] = int(depth * 255)
                    #self.assertLess(torch.abs(depth - expected_depth), 0.1, f"expected depth {expected_depth} actual {depth}")
            print(result.dtype)
            print("maxxxx:", torch.max(result))
            imageio.imwrite("recon" + image_src, result.t().fliplr().numpy())
        pass
            

if __name__ == '__main__':
    unittest.main()