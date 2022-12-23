import torch

import random

from nerf import get_camera_position, generate_rays, trace_ray


def render_image(size, transformation_matrix, fov, near, far, network, device):
    batch_size = size * size
    camera_poses = (
        get_camera_position(transformation_matrix).reshape(1, -1).repeat(batch_size, 1)
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
    render_rays(batch_size, camera_poses, rays, distance_to_depth_modifiers, near, far, network, device)


def render_rays(
    batch_size, camera_poses, rays, distance_to_depth_modifiers, near, far, network, device
):
    nears = torch.tensor(near).repeat(batch_size) / distance_to_depth_modifiers
    fars = torch.tensor(far).repeat(batch_size) / distance_to_depth_modifiers
    out_colors, dist = trace_ray(
        network.to(device),
        camera_poses.to(device),
        rays.to(device),
        100,
        nears.to(device),
        fars.to(device),
    )
    depth = dist * distance_to_depth_modifiers

    return depth, out_colors


def sample_batch(batch_size, size, transformation_matricies, images, fov):
    remaining_batch_size = batch_size
    frame_perm = torch.randperm(len(transformation_matricies))
    shuffled_transformation_matricies = transformation_matricies[frame_perm]
    shuffled_images = images[frame_perm]
    # TODO the sampling strategy is not specified and I am lazy so we're going to try something simple
    # and dumb without thinking too much
    batch_rays = []
    batch_distance_to_depth_modifiers = []
    batch_camera_poses = []
    batch_expected_colors = []
    for i, (transformation_matrix, img) in enumerate(
        zip(shuffled_transformation_matricies, shuffled_images)
    ):
        if remaining_batch_size <= 0:
            break
        chunk_size = random.randint(0, remaining_batch_size)
        if i == len(transformation_matricies) - 1:
            chunk_size = remaining_batch_size

        remaining_batch_size -= chunk_size

        chunk_camera_poses = (
            get_camera_position(transformation_matrix)
            .reshape(1, -1)
            .repeat(chunk_size, 1)
        )

        center_ray = generate_rays(
            fov, transformation_matrix[:3, :3], torch.tensor([[0.5, 0.5]]), 1
        )

        xs = torch.arange(0, size, 1)
        ys = torch.arange(0, size, 1)

        chunk_perm = torch.randperm(size * size)[0:chunk_size]

        chunk_screen_points = torch.cartesian_prod(xs / float(size), ys / float(size))[
            chunk_perm
        ] + torch.tensor([[0.5 / size, 0.5 / size]]).repeat(chunk_size, 1)

        chunk_image_pixels = torch.cartesian_prod(xs, ys)[chunk_perm]

        chunk_expected_colors = img[
            chunk_image_pixels[:, 0], chunk_image_pixels[:, 1], :
        ]

        batch_expected_colors.append(chunk_expected_colors)

        chunk_rays = generate_rays(
            fov, transformation_matrix[:3, :3], chunk_screen_points, 1
        )

        chunk_distance_to_depth_modifiers = torch.matmul(chunk_rays, center_ray.t())[
            :, 0
        ]

        batch_rays.append(chunk_rays)
        batch_distance_to_depth_modifiers.append(chunk_distance_to_depth_modifiers)
        batch_camera_poses.append(chunk_camera_poses)

    rays = torch.concat(batch_rays, 0)
    distance_to_depth_modifiers = torch.concat(batch_distance_to_depth_modifiers, 0)
    camera_poses = torch.concat(batch_camera_poses, 0)
    expected_colors = torch.concat(batch_expected_colors, 0)
    return camera_poses, rays, distance_to_depth_modifiers, expected_colors
