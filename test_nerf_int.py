import unittest
import torch
import torch.utils.benchmark as benchmark

import imageio.v3 as iio
import imageio

import os
from batch_and_sampler import render_image, render_rays, sample_batch

from nerf import load_config_file

from neural_nerf import NerfModel
from tiny_nerf import TinyNerfModel

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
        near = 0.5
        far = 7
        size = 200
        background_color = torch.tensor([0.0, 0.0, 0.0])

        fov, color_images, transformation_matricies, depth_images = load_config_file("./integration_test_data", "views", background_color, True)

        color_images /= torch.max(color_images)
        self.assertAlmostEqual(torch.max(color_images), 1.0, 3)

        num_views = color_images.shape[0]

        for view_idx in range(num_views):
            transformation_matrix = transformation_matricies[view_idx]
            expected_colors = color_images[view_idx]
            
            out_dir = "./e2e_output/test_rendering_depth_e2e_with_given_network/"

            depth, colors, _ = render_image(
                size, transformation_matrix, fov, near, far, CubeNetwork(), CubeNetwork(), 64, 128, True, "cpu", background_color
            )
            normalized_depth = (depth - near) / (far - near)
            inverted_normalized_depth = 1 - normalized_depth
            out_depth = inverted_normalized_depth * 255

            expected_depth = (1 - depth_images[view_idx].flatten() / 255.0) * (far - near) + near
            l1_depth_error = torch.abs(depth - expected_depth)
            p95_l1_depth_error = torch.quantile(l1_depth_error, 0.95)

            os.makedirs(out_dir, exist_ok=True)
            imageio.imwrite(
                out_dir + "output_depth" + str(view_idx) + ".png",
                (out_depth).reshape((size, size)).t().fliplr().numpy(),
            )
            imageio.imwrite(
                out_dir + "output_diff_" + str(view_idx)  + ".png",
                (l1_depth_error).reshape((size, size)).t().fliplr().numpy(),
            )
            imageio.imwrite(
                out_dir + "output_colors_" + str(view_idx) + ".png",
                (colors * 255)
                .reshape((size, size, 3))
                .transpose(0, 1)
                .flip([1])
                .numpy(),
            )
            imageio.imwrite(
                out_dir + "output_colors_diff" + str(view_idx) + ".png",
                (torch.abs(expected_colors.flatten() - colors.flatten()) * 255)
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
        near = 0.5
        far = 7
        background_color = torch.tensor([0.0, 0.0, 0.0])

        fov, color_images, transformation_matricies, _ = load_config_file("./integration_test_data", "views", background_color, True)
        color_images /= torch.max(color_images)

        batch_size = 4096

        camera_poses, rays, distance_to_depth_modifiers, expected_colors = sample_batch(
            batch_size,
            200,
            transformation_matricies,
            color_images,
            fov,
        )
        depth, colors, _ = render_rays(
            batch_size,
            camera_poses,
            rays,
            distance_to_depth_modifiers,
            near,
            far,
            CubeNetwork(),
            CubeNetwork(),
            64,
            128,
            True,
            "cpu",
            background_color
        )
        self.assertEqual(depth.shape, torch.Size([4096]))
        self.assertEqual(colors.shape, torch.Size([4096, 3]))
        self.assertEqual(expected_colors.shape, torch.Size([4096, 3]))

        results = torch.abs(expected_colors - colors)

        r_error = results[:, 0]
        p95_r_error = torch.quantile(r_error, 0.95)
        self.assertLess(p95_r_error, 0.005)

        b_error = results[:, 0]
        p95_b_error = torch.quantile(b_error, 0.95)
        self.assertLess(p95_b_error, 0.005)

        g_error = results[:, 0]
        p95_g_error = torch.quantile(g_error, 0.95)
        self.assertLess(p95_g_error, 0.005)

    def test_neural_nerf_render_e2e(self):
        device = "cpu"
        batch_size = 4096
        near = 0.5
        far = 7
        background_color = torch.tensor([0.0, 0.0, 0.0])

        fov, color_images, transformation_matricies = load_config_file("./integration_test_data", "views", background_color, False)
        coarse_network = NerfModel(5.0, device)
        fine_network = NerfModel(5.0, device)

        camera_poses, rays, distance_to_depth_modifiers, _ = sample_batch(
            batch_size,
            200,
            transformation_matricies,
            color_images,
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
            64,
            128,
            True,
            device,
            background_color,
        )
        self.assertEqual(depth.shape, torch.Size([4096]))
        self.assertEqual(colors.shape, torch.Size([4096, 3]))

    def test_benchmark_neural_nerf_render_e2e(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        batch_size = 4096
        near = 0.5
        far = 7

        background_color = torch.tensor([0.0, 0.0, 0.0])

        fov, color_images, transformation_matricies = load_config_file("./integration_test_data", "views", background_color, False)
        coarse_network = TinyNerfModel(5.0, device)
        fine_network = TinyNerfModel(5.0, device)

        camera_poses, rays, distance_to_depth_modifiers, _ = sample_batch(
            batch_size,
            200,
            transformation_matricies,
            color_images,
            fov,
        )

        t0 = benchmark.Timer(
            stmt='render_rays(batch_size, camera_poses, rays, distance_to_depth_modifiers, near, far, coarse_network, fine_network, 64, 128, True, device, background_color)',
            setup='from batch_and_sampler import render_rays',
            globals={
                'batch_size': batch_size,
                'camera_poses': camera_poses,
                'rays': rays,
                'distance_to_depth_modifiers': distance_to_depth_modifiers,
                'near': near,
                'far': far,
                'coarse_network': coarse_network,
                'fine_network': fine_network,
                'device': device, 
                'background_color': background_color
            }
        )
        print(t0.timeit(20))
            

    def test_profile_gpu_neural_nerf_render_e2e(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if not use_cuda:
            print("no cuda acceleration available. Skipping test")
            self.skipTest("no cuda acceleration available")

        batch_size = 500
        near = torch.tensor(0.5)
        far = torch.tensor(7.0)

        background_color = torch.tensor([0.0, 0.0, 0.0])

        fov, color_images, transformation_matricies = load_config_file("./integration_test_data", "views", background_color, False)
        coarse_network = TinyNerfModel(5.0, device)
        fine_network = TinyNerfModel(5.0, device)

        camera_poses, rays, distance_to_depth_modifiers, _ = sample_batch(
            batch_size,
            200,
            transformation_matricies,
            color_images,
            fov,
        )

        camera_poses = camera_poses.to(device)
        rays = rays.to(device)
        distance_to_depth_modifiers = distance_to_depth_modifiers.to(device)
        near = near.to(device)
        far = far.to(device)
        coarse_network = coarse_network.to(device)
        fine_network = fine_network.to(device)
        background_color = background_color.to(device)

        def trace_handler(prof):
            print(
                prof.key_averages(group_by_stack_n=5).table(
                    sort_by="cuda_time_total", row_limit=10
                )
            )
            print(
                prof.key_averages(group_by_stack_n=5).table(
                    sort_by="cpu_time_total", row_limit=10
                )
            )

        print("cuda acceleration available. Using cuda")

        for i in range(10):
            print(f"warmup run {i}")
            _, _, _ = render_rays(
                batch_size,
                camera_poses,
                rays,
                distance_to_depth_modifiers,
                near,
                far,
                coarse_network,
                fine_network,
                64,
                128,
                True,
                device,
                background_color,
            )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
            with_flops=True,
            on_trace_ready=trace_handler,
        ) as prof:
            with record_function("train_loop"):
                depth, colors, _ = render_rays(
                    batch_size,
                    camera_poses,
                    rays,
                    distance_to_depth_modifiers,
                    near,
                    far,
                    coarse_network,
                    fine_network,
                    64,
                    128,
                    True,
                    device,
                    background_color,
                )
            self.assertEqual(depth.shape, torch.Size([batch_size]))
            self.assertEqual(colors.shape, torch.Size([batch_size, 3]))
            prof.step()


if __name__ == "__main__":
    unittest.main()
