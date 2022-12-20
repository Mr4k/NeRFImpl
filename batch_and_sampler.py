import torch

import random

from nerf import get_camera_position, generate_rays, trace_ray

def render_image(size, transformation_matrix, fov, near, far, network):
    batch_size = size * size
    camera_poses = (
        get_camera_position(transformation_matrix)
        .reshape(1, -1)
        .repeat(batch_size, 1)
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
        network,
        camera_poses,
        rays,
        100,
        torch.tensor(near).repeat(batch_size) / distance_to_depth_modifiers,
        torch.tensor(far).repeat(batch_size) / distance_to_depth_modifiers,
    )
    depth = dist * distance_to_depth_modifiers

    return depth, out_colors

def render_rays(batch_size, camera_poses, rays, distance_to_depth_modifiers, near, far, network):
    out_colors, dist = trace_ray(
        network,
        camera_poses,
        rays,
        100,
        torch.tensor(near).repeat(batch_size) / distance_to_depth_modifiers,
        torch.tensor(far).repeat(batch_size) / distance_to_depth_modifiers,
    )
    depth = dist * distance_to_depth_modifiers

    return depth, out_colors

def sample_batch(batch_size, size, transformation_matricies, fov, near, far, network):
    remaining_batch_size = batch_size
    shuffled_transformation_matricies = random.shuffle(list(transformation_matricies))
    # TODO the sampling strategy is not specified and I am lazy so we're going to try something simple
    # and dumb without thinking too much
    batch_rays = []
    batch_distance_to_depth_modifiers = []
    batch_camera_poses = []
    for i, transformation_matrix in enumerate(shuffled_transformation_matricies):
        chunk_size = random.randint(remaining_batch_size)
        if i == len(transformation_matricies):
            chunk_size = remaining_batch_size
        chunk_camera_poses = (
            get_camera_position(transformation_matrix)
            .reshape(1, -1)
            .repeat(chunk_size, 1)
        )

        center_ray = generate_rays(
            fov, transformation_matrix[:3, :3], torch.tensor([[0.5, 0.5]]), 1
        )

        xs = torch.arange(0, 1, 1.0 / size)[torch.randperm(size)[0:chunk_size]]
        ys = torch.arange(0, 1, 1.0 / size)[torch.randperm(size)[0:chunk_size]]
        chunk_screen_points = torch.cartesian_prod(xs, ys) + torch.tensor(
            [[0.5 / size, 0.5 / size]]
        ).repeat(chunk_size, 1)
        
        chunk_rays = generate_rays(fov, transformation_matrix[:3, :3], chunk_screen_points, 1)

        chunk_distance_to_depth_modifiers = torch.matmul(chunk_rays, center_ray.t())[:, 0]

        batch_rays.append(chunk_rays)
        batch_distance_to_depth_modifiers.append(chunk_distance_to_depth_modifiers)
        batch_camera_poses.append(chunk_camera_poses)

    
    rays = torch.concat(batch_rays, 0)
    distance_to_depth_modifiers = torch.concat(batch_distance_to_depth_modifiers, 0)
    camera_poses = torch.concat(batch_camera_poses, 0)
    return camera_poses, rays, distance_to_depth_modifiers