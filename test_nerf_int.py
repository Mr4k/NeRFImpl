import unittest
from numpy import average
import torch

import imageio.v3 as iio
import imageio

import os
from batch_and_sampler import render_image, render_rays, sample_batch

from nerf import generate_rays, get_camera_position, load_config_file, trace_ray

from tqdm import tqdm


def cube_network(points, dirs):
    num_points, _ = points.shape

    distances_from_origin, _ = torch.max(torch.abs(points), dim=1)

    color_pyramids = [
        {
            "height_axis": 2,
            "height_axis_direction": 1,
            "color": torch.tensor([1.0, 0.0, 0.0]),
        },
        {
            "height_axis": 2,
            "height_axis_direction": -1,
            "color": torch.tensor([1.0, 0.0, 1.0]),
        },
        {
            "height_axis": 1,
            "height_axis_direction": 1,
            "color": torch.tensor([0.0, 1.0, 1.0]),
        },
        {
            "height_axis": 1,
            "height_axis_direction": -1,
            "color": torch.tensor([0.0, 1.0, 0.0]),
        },
        {
            "height_axis": 0,
            "height_axis_direction": 1,
            "color": torch.tensor([0.0, 0.0, 1.0]),
        },
        {
            "height_axis": 0,
            "height_axis_direction": -1,
            "color": torch.tensor([1.0, 1.0, 0.0]),
        },
    ]

    opacity = torch.zeros(num_points)
    opacity[distances_from_origin <= 1.0] = 10000

    colors = torch.zeros((num_points, 3))

    for pyramid in color_pyramids:
        height_axis = pyramid["height_axis"]
        height_axis_direction = pyramid["height_axis_direction"]
        color = pyramid["color"]

        other_axes = [0, 1, 2]
        other_axes.remove(height_axis)
        cond1 = points[:, height_axis] * height_axis_direction >= 0
        cond2 = torch.abs(points[:, other_axes[0]]) <= torch.abs(points[:, height_axis])
        cond3 = torch.abs(points[:, other_axes[1]]) <= torch.abs(points[:, height_axis])
        colors[cond1 & cond2 & cond3, :] = color

    return colors, opacity


class TestNerfInt(unittest.TestCase):
    def test_rendering_depth_e2e_with_given_network(self):
        frames = load_config_file("./integration_test_data/cameras.json")
        near = 0.5
        far = 7
        size = 200
        for f in frames:
            fov = torch.tensor(f["fov"])
            transformation_matrix = torch.tensor(
                f["transformation_matrix"], dtype=torch.float
            ).t()
            image_src = f["file_path"]

            fname, _ = image_src.split(".png")
            ext = "png"
            image_src = fname + "_depth" + "." + ext
            color_image_src = fname + "." + ext

            pixels = (
                torch.tensor(
                    iio.imread(os.path.join("./integration_test_data/", image_src))
                )
                .t()
                .flipud()
                .float()
            )
            out_dir = "./e2e_output/test_rendering_depth_e2e_with_given_network/"

            depth, out_colors = render_image(
                size, transformation_matrix, fov, near, far, cube_network
            )
            normalized_depth = (depth - near) / (far - near)
            inverted_normalized_depth = 1 - normalized_depth
            out_depth = inverted_normalized_depth * 255

            expected_depth = (1 - pixels.flatten() / 255.0) * (far - near) + near
            l1_depth_error = torch.abs(depth - expected_depth)
            p95_l1_depth_error = torch.quantile(l1_depth_error, 0.95)

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
                out_dir + "output_colors_" + color_image_src,
                (out_colors * 255)
                .reshape((size, size, 3))
                .transpose(0, 1)
                .flip([1])
                .numpy(),
            )

            expected_p95_l1_depth_error = 0.1
            self.assertLess(
                p95_l1_depth_error,
                expected_p95_l1_depth_error,
                f"expected p95 depth error {p95_l1_depth_error} to be smaller than {expected_p95_l1_depth_error}",
            )

    def test_neural_nerf_render_e2e(self):
        frames = load_config_file("./integration_test_data/cameras.json")

        batch_size = 4096

        transformation_matricies = []
        for f in frames:
            fov = torch.tensor(f["fov"])
            transformation_matrix = torch.tensor(
                f["transformation_matrix"], dtype=torch.float
            ).t()
            transformation_matricies.append(transformation_matrix)
        camera_poses, rays, distance_to_depth_modifiers = sample_batch(
            batch_size, 200, transformation_matricies, 45 * torch.pi / 180
        )
        depth, color = render_rays(
            batch_size, camera_poses, rays, distance_to_depth_modifiers
        )
        pass


if __name__ == "__main__":
    unittest.main()
