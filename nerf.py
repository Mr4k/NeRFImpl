import torch
import hydra

import hydra
from hydra.core.hydra_config import HydraConfig

from wandb_wrapper import wandb_init, wandb_log

"""
Returns an arr of n sampling points
"""
def compute_stratified_sample_points(n, t_near, t_far):
    bin_width = t_far - t_near
    return t_near + bin_width * (torch.rand(n) + torch.arange(n)) / n

"""
Returns a tuple (colors, opacity)
"""
def get_network_output(network, points, dirs):
    pass

def trace_ray(network, pos, dir, n, t_near, t_far):
    # TODO hmmm n + 1?
    stratified_sample_points = compute_stratified_sample_points(n + 1, t_near, t_far)
    colors, opacity = get_network_output(network, stratified_sample_points * dir + pos, dir)

    cum_partial_passthrough_sum = 0
    cum_color = torch.zeros(3)

    for i in range(n):
        delta = stratified_sample_points[i + 1] - stratified_sample_points[i]
        prob_hit_current_bin = (1 - torch.exp(opacity[i]*delta))
        cum_passthrough_prob = torch.exp(-cum_partial_passthrough_sum)
        cum_color = cum_passthrough_prob * prob_hit_current_bin * colors[i]
        cum_partial_passthrough_sum += opacity[i] * delta
    return cum_color

@hydra.main(version_base=None, config_path=".", config_name="config")
def nerf_main(cfg):
    if cfg.wandb.enabled:
        wandb_init(cfg)
    wandb_log({"hello": 1})
    torch.save(torch.rand((2,2)), f"{HydraConfig.get().runtime.output_dir}/out.pth")

if __name__ == "__main__":
    nerf_main()
