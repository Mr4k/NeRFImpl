import unittest
import torch

import imageio.v3 as iio
import imageio

import os
from batch_and_sampler import render_image, render_rays, sample_batch

from nerf import load_config_file

from neural_nerf import NerfModel

from torch.profiler import profile, record_function, ProfilerActivity

class CubeNetwork:
    def to(self, _):
        return self

    def __call__(self, points, dirs):
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
            cond2 = torch.abs(points[:, other_axes[0]]) <= torch.abs(
                points[:, height_axis]
            )
            cond3 = torch.abs(points[:, other_axes[1]]) <= torch.abs(
                points[:, height_axis]
            )
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
                torch.tensor(iio.imread(os.path.join("./integration_test_data/", image_src)))
                .t()
                .flipud()
                .float()
            )
            color_pixels = (
                torch.tensor(iio.imread(os.path.join("./integration_test_data/", color_image_src)))[
                    :, :, :3
                ]
                .transpose(0, 1)
                .flip([0])
                .float()
                / 255.0
            )
            color_pixels /= torch.max(color_pixels)
            self.assertAlmostEqual(torch.max(color_pixels), 1.0, 3)

            out_dir = "./e2e_output/test_rendering_depth_e2e_with_given_network/"

            depth, out_colors = render_image(
                size, transformation_matrix, fov, near, far, CubeNetwork(), "cpu"
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
                (torch.abs(color_pixels.flatten() - out_colors.flatten()) * 255)
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
        frames = load_config_file("./integration_test_data/cameras.json")

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
            pixels = (
                torch.tensor(iio.imread(os.path.join("./integration_test_data/", image_src)))[:, :, :3]
                .transpose(0, 1)
                .flip([0])
                .float()
                / 255.0
            )

            pixels /= torch.max(pixels)
            self.assertAlmostEqual(torch.max(pixels), 1.0, 3)

            images.append(pixels)

        camera_poses, rays, distance_to_depth_modifiers, expected_colors = sample_batch(
            batch_size,
            200,
            torch.stack(transformation_matricies),
            torch.stack(images),
            fov,
        )
        depth, colors = render_rays(
            batch_size,
            camera_poses,
            rays,
            distance_to_depth_modifiers,
            near,
            far,
            CubeNetwork(),
            "cpu",
        )
        self.assertEqual(depth.shape, torch.Size([4096]))
        self.assertEqual(colors.shape, torch.Size([4096, 3]))
        self.assertEqual(expected_colors.shape, torch.Size([4096, 3]))

        results = torch.abs(expected_colors - colors)

        r_error = results[:, 0]
        p95_r_error = torch.quantile(r_error, 0.97)
        self.assertLess(p95_r_error, 0.005)

        b_error = results[:, 0]
        p95_b_error = torch.quantile(b_error, 0.97)
        self.assertLess(p95_b_error, 0.005)

        g_error = results[:, 0]
        p95_g_error = torch.quantile(g_error, 0.97)
        self.assertLess(p95_g_error, 0.005)

    def _load_examples_from_config(self):
        config = load_config_file("./integration_test_data/transforms_views.json")
        transformation_matricies = []
        images = []
        fov = torch.tensor(config["camera_angle_x"])
        for f in config["frames"]:
            transform_matrix = torch.tensor(
                f["transform_matrix"], dtype=torch.float
            ).t()
            transformation_matricies.append(transform_matrix)
            image_src = f["file_path"] + ".png"
            pixels = (
                torch.tensor(iio.imread(os.path.join("./integration_test_data/", image_src)))[:, :, :3]
                .transpose(0, 1)
                .flip([0])
                .float()
                / 255.0
            )

            pixels /= torch.max(pixels)
            self.assertAlmostEqual(torch.max(pixels), 1.0, 3)

            images.append(pixels)
            return fov, torch.stack(transformation_matricies), torch.stack(images)

    def test_neural_nerf_render_e2e(self):
        device = "cpu"
        batch_size = 4096
        near = 0.5
        far = 7

        fov, transform_matricies, images = self._load_examples_from_config()
        coarse_network = NerfModel(5.0, device)
        fine_network = NerfModel(5.0, device)

        camera_poses, rays, distance_to_depth_modifiers, _ = sample_batch(
            batch_size,
            200,
            transform_matricies,
            images,
            fov,
        )
        depth, colors, _ = render_rays(
            batch_size,
            camera_poses,
            rays,
            distance_to_depth_modifiers,
            near,
            far,
            coarse_network,
            fine_network,
            device
        )
        self.assertEqual(depth.shape, torch.Size([4096]))
        self.assertEqual(colors.shape, torch.Size([4096, 3]))

    def test_profile_gpu_neural_nerf_render_e2e(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if not use_cuda:
            print("no cuda acceleration available. Skipping test")
            self.skipTest("no cuda acceleration available")

        batch_size = 3000
        near = 0.5
        far = 7

        fov, transform_matricies, images = self._load_examples_from_config()
        coarse_network = NerfModel(5.0, device)
        fine_network = NerfModel(5.0, device)

        print("cuda acceleration available. Using cuda")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
            with record_function("train_loop"):                
                camera_poses, rays, distance_to_depth_modifiers, _ = sample_batch(
                    batch_size,
                    200,
                    transform_matricies,
                    images,
                    fov,
                )
                depth, colors, _ = render_rays(
                    batch_size,
                    camera_poses,
                    rays,
                    distance_to_depth_modifiers,
                    near,
                    far,
                    coarse_network,
                    fine_network,
                    device
                )
        self.assertEqual(depth.shape, torch.Size([batch_size]))
        self.assertEqual(colors.shape, torch.Size([batch_size, 3]))
        print(prof.key_averages(group_by_stack_n=10).table(sort_by="cpu_time_total", row_limit=10))


if __name__ == "__main__":
    unittest.main()
