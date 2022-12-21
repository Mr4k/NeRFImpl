import unittest
from numpy import average
import torch

import imageio.v3 as iio
import imageio

import os
from batch_and_sampler import render_image, render_rays, sample_batch

from nerf import load_config_file

from tqdm import tqdm

from neural_nerf import NerfModel


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
        frames = load_config_file("./idata/cameras.json")
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
                    iio.imread(os.path.join("./idata/", image_src))
                )
                .t()
                .flipud()
                .float()
            )
            color_pixels = (
                torch.tensor(
                    iio.imread(os.path.join("./idata/", color_image_src))
                )[:, :, :3]
                .transpose(0, 1)
                .flip([0])
                .float() / 255.0
            )
            print("shaaaap:", color_pixels.shape)

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
            imageio.imwrite(
                out_dir + "output_colors_diff" + color_image_src,
                (color_pixels * 255)
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

    def test_sample_batch_nerf_render_e2e(self):
        frames = load_config_file("./idata/cameras.json")

        batch_size = 4096

        near = 0.5
        far = 7

        transformation_matricies = []
        images = []
        for f in frames:
            fov = torch.tensor(f["fov"])
            transformation_matrix = torch.tensor(
                f["transformation_matrix"], dtype=torch.float
            ).t()
            transformation_matricies.append(transformation_matrix)
            image_src = f["file_path"]
            pixels = torch.tensor(
                    iio.imread(os.path.join("./idata/", image_src))
                )[:, :, :3].transpose(0, 1).flip([0]).float() / 255.0

            #pixels = torch.rand(pixels.shape)
            images.append(pixels)
            
        camera_poses, rays, distance_to_depth_modifiers, expected_colors = sample_batch(
            batch_size, 200, torch.stack(transformation_matricies), torch.stack(images), fov
        )
        depth, colors = render_rays(
            batch_size, camera_poses, rays, distance_to_depth_modifiers, near, far, cube_network
        )
        self.assertEqual(depth.shape, torch.Size([4096]))
        self.assertEqual(colors.shape, torch.Size([4096, 3]))
        self.assertEqual(expected_colors.shape, torch.Size([4096, 3]))

        results = torch.abs(expected_colors - colors)

        r_error = results[:, 0]
        p95_r_error = torch.quantile(r_error, 0.85)
        self.assertLess(p95_r_error, 0.005)

        b_error = results[:, 0]
        p95_b_error = torch.quantile(b_error, 0.85)
        self.assertLess(p95_b_error, 0.005)

        g_error = results[:, 0]
        p95_g_error = torch.quantile(g_error, 0.85)
        self.assertLess(p95_g_error, 0.005)

    def test_neural_nerf_render_e2e(self):
        frames = load_config_file("./idata/cameras.json")

        batch_size = 4096

        near = 0.5
        far = 7

        transformation_matricies = []
        for f in frames:
            fov = torch.tensor(f["fov"])
            transformation_matrix = torch.tensor(
                f["transformation_matrix"], dtype=torch.float
            ).t()
            transformation_matricies.append(transformation_matrix)

        camera_poses, rays, distance_to_depth_modifiers = sample_batch(
            batch_size, 200, transformation_matricies, fov
        )
        model = NerfModel()
        depth, colors = render_rays(
            batch_size, camera_poses, rays, distance_to_depth_modifiers, near, far, model.forward
        )
        self.assertEqual(depth.shape, torch.Size([4096]))
        self.assertEqual(colors.shape, torch.Size([4096, 3]))


if __name__ == "__main__":
    unittest.main()
