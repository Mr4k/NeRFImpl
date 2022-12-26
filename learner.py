import uuid
import torch
import wandb
from batch_and_sampler import render_image, render_rays, sample_batch
from matrix_math_utils import (
    generate_random_gimbal_transformation_matrix,
    generate_rot_x,
    generate_rot_z,
    generate_translation,
)
from nerf import load_config_file

from neural_nerf import NerfModel

import imageio.v3 as iio
import imageio

import os

from wandb_wrapper import wandb_init, wandb_log


def train():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        print("cuda acceleration available. Using cuda")

    wandb_init({"entity": "mr4k", "project": "nerf"})

    training_run_id = uuid.uuid4()
    out_dir = f"./training_output/runs/{training_run_id}/"
    train_data_path = "./data/lego/train"

    print(f"creating output dir: {out_dir}")
    os.makedirs(out_dir)

    # (batch size, 3)
    scale = 5.0
    batch_size = 4096

    config = load_config_file(os.path.join(train_data_path, "transforms_train.json"))
    transformation_matricies = []
    images = []
    fov = None
    if "camera_angle_x" in config:
        fov = torch.tensor(config["camera_angle_x"])
    for f in config["frames"]:
        if "fov" in f:
            fov = torch.tensor(f["fov"])
        transformation_matrix = torch.tensor(
            f["transform_matrix"], dtype=torch.float
        ).t()
        transformation_matricies.append(transformation_matrix)
        image_src = f["file_path"] + ".png"
        pixels = (
            torch.tensor(iio.imread(os.path.join(train_data_path, image_src)))[:, :, :3]
            .transpose(0, 1)
            .flip([0])
            .float()
            / 255.0
        )

        pixels /= torch.max(pixels)
        width, height, channels = pixels.shape

        assert width == height
        assert channels == 3

        images.append(pixels)

    near = 0.5
    far = 7

    loss_fn = lambda outputs, labels: ((outputs - labels) ** 2).sum()
    model = NerfModel(scale, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    num_steps = 100000
    num_steps_to_render = 100
    novel_view_transformation_matricies = [
        generate_random_gimbal_transformation_matrix(scale) for _ in range(10)
    ]
    for step in range(num_steps):
        if step % num_steps_to_render == 0:
            for i, novel_view_transformation_matrix in enumerate(
                novel_view_transformation_matricies
            ):
                print(f"rendering snapshot from view {i} at step {step}")
                size = 200

                with torch.no_grad():
                    depth_image, color_image = render_image(
                        size,
                        novel_view_transformation_matrix,
                        fov,
                        near,
                        far,
                        model,
                        device,
                    )

                out_depth_image = (
                    1.0
                    - (
                        (depth_image.cpu().detach())
                        .reshape((size, size))
                        .t()
                        .fliplr()
                        .numpy()
                        - near
                    )
                    / (far - near)
                ) * 255
                out_color_image = (
                    (color_image.cpu().detach() * 255)
                    .reshape((size, size, 3))
                    .transpose(0, 1)
                    .flip([1])
                    .numpy()
                )
                imageio.imwrite(
                    out_dir + f"view_{i}_depth_step_{step}.png", out_depth_image
                )
                imageio.imwrite(
                    out_dir + f"view_{i}_color_step_{step}.png",
                    out_color_image,
                )
                print("saved snapshot")

        optimizer.zero_grad()

        camera_poses, rays, distance_to_depth_modifiers, expected_colors = sample_batch(
            batch_size,
            200,
            torch.stack(transformation_matricies),
            torch.stack(images),
            fov,
        )

        optimizer.zero_grad()

        _, colors = render_rays(
            batch_size,
            camera_poses,
            rays,
            distance_to_depth_modifiers,
            near,
            far,
            model,
            device,
        )

        loss = loss_fn(colors.flatten(), expected_colors.flatten())
        loss.backward()

        optimizer.step()
        loss_at_step = loss.item()
        print(f"loss at step {step}: {loss_at_step}")
        wandb_log({"loss": loss_at_step, "step": step})


if __name__ == "__main__":
    train()
