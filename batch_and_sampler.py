import torch

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