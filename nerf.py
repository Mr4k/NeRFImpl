import json
import torch
import hydra

import hydra
from hydra.core.hydra_config import HydraConfig

from wandb_wrapper import wandb_init, wandb_log

"""
Returns an arr of n sampling points
"""


def compute_stratified_sample_points(batch_size, n, t_near, t_far):
    bin_width = t_far - t_near
    return (
        torch.tensor(t_near).repeat(batch_size, 2)
        + bin_width
        * (torch.rand((batch_size, n)) + torch.arange(n).repeat(batch_size, 1))
        / n
    )


"""
Returns a tuple (colors, opacity)
"""


def get_network_output(network, points, dirs):
    return network(points, dirs)


def trace_ray(network, positions, directions, n, t_near, t_far):
    # TODO hmmm n + 1?
    batch_size = positions.shape[0]

    assert positions.shape[0] == directions.shape[0]

    stratified_sample_times = compute_stratified_sample_points(n + 1, t_near, t_far)
    stratified_sample_points_centered_at_the_origin = (
        stratified_sample_times.reshape(-1, 1).repeat(1, 3) * directions
    )
    stratified_sample_points = (
        stratified_sample_points_centered_at_the_origin
        + positions.reshape(-1, 1).repeat(1, n + 1).t()
    )
    colors, opacity = get_network_output(network, stratified_sample_points, directions)

    cum_partial_passthrough_sum = torch.tensor(0.0)
    cum_color = torch.zeros(3)
    cum_expected_distance = torch.tensor(0.0)
    distance_acc = torch.tensor(t_near)

    for i in range(n):
        delta = stratified_sample_times[i + 1] - stratified_sample_times[i]
        prob_hit_current_bin = 1 - torch.exp(-opacity[i] * delta)
        cum_passthrough_prob = torch.exp(-cum_partial_passthrough_sum)

        cum_color = cum_passthrough_prob * prob_hit_current_bin * colors[i]

        # we assume probability of collision is uniform in the bin
        # TODO this might be an overestimate by delta / 2
        curr_distance = distance_acc + delta / 2

        cum_expected_distance += (
            cum_passthrough_prob * prob_hit_current_bin * curr_distance
        )

        cum_partial_passthrough_sum += opacity[i] * delta
        distance_acc += delta

    # add far plane
    cum_passthrough_prob = torch.exp(-cum_partial_passthrough_sum)
    cum_expected_distance += cum_passthrough_prob * t_far

    return cum_color, torch.min(
        torch.max(cum_expected_distance, torch.tensor(t_near)), torch.tensor(t_far)
    )


def load_config_file(path):
    with open(path) as f:
        config = json.load(f)
        if "frames" not in config:
            raise Exception(
                f"keyword 'frames' not found in config file at path: {path}"
            )
        return config["frames"]
    pass


"""
fov: angle of field of view in radians
camera_transformation_matrix: a 3x3 matrix representing the rotation of the camera
px, py: the locations of the pixels on the screen in [0, 1]
aspect_ratio: width / height
"""


def generate_ray(fov, camera_rotation_matrix, px, py, aspect_ratio=1):
    # opp / adj
    # adj = 1
    # opp = torch.tan(fov / 2.0)
    x = (2 * px - 1) * torch.tan(fov / 2.0) * aspect_ratio
    y = (2 * py - 1) * torch.tan(fov / 2.0)
    z = 1.0
    ray = torch.tensor([[x, y, z]])
    ray = torch.mm(ray, camera_rotation_matrix)
    ray /= ray.norm()
    return -ray.flatten()


def get_camera_position(camera_transformation_matrix):
    ray = torch.tensor([[0, 0, 0, 1.0]])
    return torch.mm(ray, camera_transformation_matrix)[0, :3]


@hydra.main(version_base=None, config_path=".", config_name="config")
def nerf_main(cfg):
    if cfg.wandb.enabled:
        wandb_init(cfg)
    wandb_log({"hello": 1})
    torch.save(torch.rand((2, 2)), f"{HydraConfig.get().runtime.output_dir}/out.pth")


if __name__ == "__main__":
    nerf_main()
