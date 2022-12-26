import json
import torch

from wandb_wrapper import wandb_init, wandb_log


def inverse_transform_sampling(stopping_probs, bin_boundary_points, n):
    num_batches, _ = stopping_probs.shape
    sample_bins = torch.multinomial(stopping_probs, n, True)
    indexes = torch.arange(0, num_batches, 1).repeat_interleave(n)
    flatten_samples = sample_bins.flatten()
    return (
        (
            bin_boundary_points[indexes, flatten_samples + 1]
            - bin_boundary_points[indexes, flatten_samples]
        )
        * torch.rand((num_batches * n))
        + bin_boundary_points[indexes, flatten_samples]
    ).view(num_batches, -1)


"""
Returns an arr of n sampling points

batch_size: int
n: int
t_near: tensor dims = (batch_size)
t_far: tensor dims = (batch_size)
"""


def compute_stratified_sample_points(batch_size, n, t_near, t_far):
    bin_width = t_far - t_near
    return (
        torch.tensor(t_near).reshape(-1, 1).repeat(1, n)
        + bin_width.reshape(-1, 1)
        * (torch.rand((batch_size, n)) + torch.arange(n).repeat(batch_size, 1))
        / n
    )


"""
Returns a tuple (colors, opacity)
"""


def get_network_output(network, points, dirs):
    return network(points, dirs)


"""
network: TBD
positions: tensor dims = (batch_size, 3)
directions: tensor dims = (batch_size, 3)
n: int
t_near: tensor dims = (batch_size)
t_far: tensor dims = (batch_size)
"""


def trace_ray(device, network, positions, directions, n, t_near, t_far):
    # TODO hmmm n + 1?
    batch_size = positions.shape[0]

    assert positions.shape[0] == directions.shape[0]
    assert t_near.shape[0] == batch_size
    assert len(t_near.shape) == 1
    assert t_far.shape[0] == batch_size
    assert len(t_far.shape) == 1

    stratified_sample_times = compute_stratified_sample_points(
        batch_size, n + 1, t_near, t_far
    )

    stratified_sample_points_centered_at_the_origin = stratified_sample_times.reshape(
        batch_size, n + 1, 1
    ).repeat(1, 1, 3) * directions.reshape(batch_size, 1, -1).repeat(1, n + 1, 1)
    stratified_sample_points = (
        stratified_sample_points_centered_at_the_origin
        + positions.reshape(batch_size, 1, -1).repeat(1, n + 1, 1)
    )

    # tiny cuda region at first
    colors, opacity = get_network_output(
        network.to(device),
        stratified_sample_points.view(-1, 3).to(device),
        directions.repeat_interleave(n + 1, dim=0).to(device),
    )
    colors = colors.cpu().reshape(batch_size, n + 1, 3)
    opacity = opacity.cpu().reshape(batch_size, n + 1)

    cum_partial_passthrough_sum = torch.zeros(batch_size)

    # a tensor that gives the probablity of the ray terminating in the nth bin
    # note this is really only need for the course network
    # might want to refactor the code so it can be disabled for the fine network
    stopping_probs = torch.zeros((batch_size, n))
    cum_color = torch.zeros((batch_size, 3))
    cum_expected_distance = torch.zeros(batch_size)
    distance_acc = torch.ones(batch_size) * t_near

    # TODO (getting rid of this for loop likely speeds up rendering)
    # on second thought maybe not, bottleneck will eventually likely be get_network_output
    for i in range(n):
        delta = stratified_sample_times[:, i + 1] - stratified_sample_times[:, i]
        prob_hit_current_bin = 1 - torch.exp(-opacity[:, i] * delta)
        cum_passthrough_prob = torch.exp(-cum_partial_passthrough_sum)

        stopping_probs[:, i] = cum_passthrough_prob * prob_hit_current_bin

        cum_color += stopping_probs[:, i].reshape(-1, 1).repeat(1, 3) * colors[:, i]

        cum_expected_distance += stopping_probs[:, i] * distance_acc

        cum_partial_passthrough_sum += opacity[:, i] * delta
        distance_acc += delta

    # add far plane
    cum_passthrough_prob = torch.exp(-cum_partial_passthrough_sum)
    cum_expected_distance += cum_passthrough_prob * t_far

    return (
        cum_color,
        torch.min(
            torch.max(cum_expected_distance, torch.tensor(t_near)), torch.tensor(t_far)
        ),
        stopping_probs,
    )


def load_config_file(path):
    with open(path) as f:
        config = json.load(f)
        if "frames" not in config:
            raise Exception(
                f"keyword 'frames' not found in config file at path: {path}"
            )
        return config
    pass


"""
fov: angle of field of view in radians
# TODO note this is super funky, assuming orbiting camera setup and returns (-) direction camera rotation matrix should be pointing it in
# Fix this whole system to work with generic cameras
camera_transformation_matrix: a 3x3 matrix representing the rotation of the camera
screen_points: (batch_size, 2), the locations of the pixels on the screen in [0, 1]^2
aspect_ratio: width / height

output: (batch_size, 3)
"""


def generate_rays(fov, camera_rotation_matrix, screen_points, aspect_ratio=1):
    # opp / adj
    # adj = 1
    # opp = torch.tan(fov / 2.0)
    batch_size = screen_points.shape[0]
    x = (2 * screen_points[:, 0] - 1) * torch.tan(fov / 2.0) * aspect_ratio
    y = (2 * screen_points[:, 1] - 1) * torch.tan(fov / 2.0)
    z = torch.tensor([1.0]).repeat(batch_size)
    ray = torch.stack([x, y, z], dim=-1)
    ray = torch.mm(ray, camera_rotation_matrix)
    ray /= ray.norm(dim=1).reshape(-1, 1).repeat(1, 3)
    return -ray


def get_camera_position(camera_transformation_matrix):
    ray = torch.tensor([[0, 0, 0, 1.0]])
    return torch.mm(ray, camera_transformation_matrix)[0, :3]
