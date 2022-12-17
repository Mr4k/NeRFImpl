import unittest
from numpy import average
import torch

import imageio.v3 as iio
import imageio

import os

from nerf import generate_rays, get_camera_position, load_config_file, trace_ray

from tqdm import tqdm

def cube_network(points, dirs):
    batch_size, num_points, _ = points.shape

    distances_from_origin, _ = torch.max(torch.abs(points), dim=2)

    color_pyramids = [
        {
            "height_axis": 2,
            "height_axis_direction": 1,
            "color": torch.tensor([1.0, 0.0, 0.0])
        },
        {
            "height_axis": 2,
            "height_axis_direction": -1,
            "color": torch.tensor([1.0, 0.0, 1.0])
        },
        {
            "height_axis": 1,
            "height_axis_direction": 1,
            "color": torch.tensor([0.0, 1.0, 1.0])
        },
        {
            "height_axis": 1,
            "height_axis_direction": -1,
            "color": torch.tensor([0.0, 1.0, 0.0])
        },
        {
            "height_axis": 0,
            "height_axis_direction": 1,
            "color": torch.tensor([1.0, 1.0, 0.0])
        },
        {
            "height_axis": 0,
            "height_axis_direction": -1,
            "color": torch.tensor([1.0, 1.0, 0.0])
        },
    ]

    opacity = torch.zeros((batch_size, num_points))
    opacity[distances_from_origin <= 1.0] = 10000

    colors = torch.zeros((batch_size, num_points, 3))

    """for pyramid in color_pyramids:
        height_axis = pyramid["height_axis"]
        height_axis_direction = pyramid["height_axis_direction"]
        color = pyramid["color"]

        other_axes = [0, 1, 2]
        other_axes.remove(height_axis)
        cond1 = points[:, :, height_axis] * height_axis_direction >= 0
        cond2 = torch.abs(points[:, :, other_axes[0]] - points[:, :, height_axis]) <= 1
        cond3 = torch.abs(points[:, :, other_axes[1]] - points[:, :, height_axis]) <= 1
        colors[cond1 & cond2 & cond3, :] = color"""
    sphere_ds = torch.norm(points, dim = 2)
    colors[sphere_ds < 1.2, :] = torch.tensor([1.0, 1.0, 0.0])

    return colors, opacity

class TestNerfUnit(unittest.TestCase):
    def test_rendering_depth_e2e_with_given_network(self):
        frames = load_config_file("./integration_test_data/cameras.json")
        near = 0.5
        far = 7
        for f in frames:
            fov = torch.tensor(f["fov"])
            #fov = torch.tensor(10 / 180 * torch.pi)
            transformation_matrix = torch.tensor(
                f["transformation_matrix"], dtype=torch.float
            ).t()
            image_src = f["file_path"]

            size = 200
            batch_size = size * size
            camera_poses = (
                get_camera_position(transformation_matrix)
                .reshape(1, -1)
                .repeat(batch_size, 1)
            )

            fname, _ = image_src.split(".png")
            ext = "png"
            image_src = fname + "_depth" + "." + ext

            pixels = (
                torch.tensor(
                    iio.imread(os.path.join("./integration_test_data/", image_src))
                )
                .t()
                .flipud()
                .float()
            )

            center_ray = generate_rays(
                fov, transformation_matrix[:3, :3], torch.tensor([[0.5, 0.5]]), 1
            )

            xs = torch.arange(0, 1, 1.0 / size)
            ys = torch.arange(0, 1, 1.0 / size)
            screen_points = torch.cartesian_prod(xs, ys) + torch.tensor(
                [[0.5 / size, 0.5 / size]]
            ).repeat(batch_size, 1)

            rays = generate_rays(fov, transformation_matrix[:3, :3], screen_points, 1)
            distance_to_depth_modifiers = torch.matmul(rays, center_ray.t())[:, 0]
            out_colors, dist = trace_ray(
                cube_network,
                camera_poses,
                rays,
                100,
                torch.tensor(near).repeat(batch_size) / distance_to_depth_modifiers,
                torch.tensor(far).repeat(batch_size) / distance_to_depth_modifiers,
            )
            depth = dist * distance_to_depth_modifiers
            normalized_depth = (depth - near) / (far - near)
            inverted_normalized_depth = 1 - normalized_depth
            out_depth = inverted_normalized_depth * 255
            out_dir = "./e2e_output/test_rendering_depth_e2e_with_given_network/"

            expected_depth = (1 - pixels.flatten() / 255.0) * (far - near) + near
            print("max", torch.min(expected_depth), torch.max(expected_depth))
            l1_depth_error = torch.abs(depth - expected_depth)
            p95_l1_depth_error = torch.quantile(l1_depth_error, 0.95)
            print("p95 min:", torch.min(p95_l1_depth_error))

            os.makedirs(out_dir, exist_ok=True)
            imageio.imwrite(
                out_dir + "output_" + image_src,
                (out_depth).reshape((size, size)).t().fliplr().numpy(),
            )
            imageio.imwrite(
                out_dir + "output_diff_" + image_src,
                (l1_depth_error).reshape((size, size)).t().fliplr().numpy(),
            )

            imageio.imwrite(
                out_dir + "output_colors_" + image_src,
                (out_colors * 255).reshape((size, size, 3)).transpose(0, 1).flip([1]).numpy(),
            )

            expected_p95_l1_depth_error = 0.1
            self.assertLess(
                p95_l1_depth_error,
                expected_p95_l1_depth_error,
                f"expected p95 depth error {p95_l1_depth_error} to be smaller than {expected_p95_l1_depth_error}",
            )


if __name__ == "__main__":
    unittest.main()
