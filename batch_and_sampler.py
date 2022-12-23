from numpy import dtype
import torch

from torch.distributions import Categorical

import random

from nerf import get_camera_position, generate_rays, trace_ray


def render_image(size, transformation_matrix, fov, near, far, network, device):
    total_rays = size * size
    batch_size = 2048
    camera_poses = (
        get_camera_position(transformation_matrix).reshape(1, -1).repeat(total_rays, 1)
    )

    center_ray = generate_rays(
        fov, transformation_matrix[:3, :3], torch.tensor([[0.5, 0.5]]), 1
    )

    xs = torch.arange(0, 1, 1.0 / size)
    ys = torch.arange(0, 1, 1.0 / size)
    screen_points = torch.cartesian_prod(xs, ys) + torch.tensor(
        [[0.5 / size, 0.5 / size]]
    ).repeat(total_rays, 1)
    

    rays = generate_rays(fov, transformation_matrix[:3, :3], screen_points, 1)
    distance_to_depth_modifiers = torch.matmul(rays, center_ray.t())[:, 0]

    ray_batches = rays.split(batch_size)
    distance_to_depth_modifiers_batches = distance_to_depth_modifiers.split(batch_size)
    camera_pose_batches = camera_poses.split(batch_size)

    batch_depths = []
    batch_colors = []
    for ray_batch, ddm_batch, camera_pos_batch in zip(ray_batches, distance_to_depth_modifiers_batches, camera_pose_batches):
        num_rays = ray_batch.shape[0]
        depths, colors = render_rays(num_rays, camera_pos_batch.clone(), ray_batch.clone(), ddm_batch.clone(), near, far, network, device)
        batch_depths.append(depths)
        batch_colors.append(colors)

    return torch.concat(batch_depths, dim=0), torch.concat(batch_colors, dim=0)


def render_rays(
    batch_size, camera_poses, rays, distance_to_depth_modifiers, near, far, network, device
):
    nears = torch.tensor(near).repeat(batch_size) / distance_to_depth_modifiers
    fars = torch.tensor(far).repeat(batch_size) / distance_to_depth_modifiers
    out_colors, dist = trace_ray(
        device,
        network,
        camera_poses,
        rays,
        100,
        nears,
        fars,
    )
    depth = dist * distance_to_depth_modifiers

    return depth, out_colors

def random_partition(num_catagories, num_draws):
    cat = Categorical(torch.ones(num_catagories)/float(num_catagories))
    return torch.histogram(cat.sample_n(num_draws).to(torch.float), bins=int(num_catagories))[0].to(torch.int)

def sample_batch(batch_size, size, transformation_matricies, images, fov):
    frame_perm = torch.randperm(len(transformation_matricies))
    shuffled_transformation_matricies = transformation_matricies[frame_perm]
    shuffled_images = images[frame_perm]
    # TODO the sampling strategy is not specified and I am lazy so we're going to try something simple
    # and dumb without thinking too much
    batch_rays = []
    batch_distance_to_depth_modifiers = []
    batch_camera_poses = []
    batch_expected_colors = []
    chunk_sizes = random_partition(len(transformation_matricies), batch_size)
    for i, (transformation_matrix, img, chnk_size) in enumerate(
        zip(shuffled_transformation_matricies, shuffled_images, chunk_sizes)
    ):
        chnk_size = chnk_size.item()
        chunk_camera_poses = (
            get_camera_position(transformation_matrix)
            .reshape(1, -1)
            .repeat(chnk_size, 1)
        )

        center_ray = generate_rays(
            fov, transformation_matrix[:3, :3], torch.tensor([[0.5, 0.5]]), 1
        )

        xs = torch.arange(0, size, 1)
        ys = torch.arange(0, size, 1)

        chunk_perm = torch.randperm(size * size)[0:chnk_size]

        chunk_screen_points = torch.cartesian_prod(xs / float(size), ys / float(size))[
            chunk_perm
        ] + torch.tensor([[0.5 / size, 0.5 / size]]).repeat(chnk_size, 1)

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
