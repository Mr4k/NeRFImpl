import torch
from batch_and_sampler import render_rays, sample_batch
from matrix_math_utils import generate_rot_x, generate_translation
from nerf import load_config_file

from neural_nerf import NerfModel

import imageio.v3 as iio
import imageio

import os

def create_novel_gimbal_transformation_matrix(scale):
    # TODO this function
    pass

def train():
    # (batch size, 3) 
    scale = 1.0 / 5.0
    batch_size = 4096

    frames = load_config_file("./data/cube/train/transforms_test.json")
    transformation_matricies = []
    images = []
    fov = None
    for f in frames:
        fov = torch.tensor(f["fov"])
        transformation_matrix = torch.tensor(
            f["transformation_matrix"], dtype=torch.float
        ).t()
        transformation_matricies.append(transformation_matrix)
        image_src = f["file_path"]
        pixels = torch.tensor(
                iio.imread(os.path.join("./data/cube/train/", image_src))
            )[:, :, :3].transpose(0, 1).flip([0]).float() / 255.0

        pixels /= torch.max(pixels)
        images.append(pixels)

    near = 0.5
    far = 7

    loss_fn = lambda outputs, labels: ((outputs - labels) ** 2).sum()
    model = NerfModel(scale)

    optimizer = torch.optim.Adam(model.parameters(), 0.0005)

    num_steps = 100000
    for step in range(num_steps):
        camera_poses, rays, distance_to_depth_modifiers, expected_colors = sample_batch(
            batch_size, 200, torch.stack(transformation_matricies), torch.stack(images), fov
        )

        optimizer.zero_grad()

        _, colors = render_rays(
            batch_size, camera_poses, rays, distance_to_depth_modifiers, near, far, model
        )

        loss = loss_fn(colors.flatten(), expected_colors.flatten())
        loss.backward()

        optimizer.step()
        print(f"loss at step {step}: {loss.item()}")

if __name__ == "__main__":
    train()
