import uuid
import torch
import wandb
from batch_and_sampler import render_image, render_rays, sample_batch
from matrix_math_utils import generate_random_gimbal_transformation_matrix, generate_rot_x, generate_rot_z, generate_translation
from nerf import load_config_file

from neural_nerf import NerfModel

import imageio.v3 as iio
import imageio

import os

from wandb_wrapper import wandb_init, wandb_log


def train():
    wandb_init({
        "entity": "mr4k",
        "project": "nerf"
    })

    training_run_id = uuid.uuid4()
    out_dir = f"./training_output/runs/{training_run_id}/"

    print(f"creating output dir: {out_dir}")
    os.makedirs(out_dir)

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
        pixels = (
            torch.tensor(iio.imread(os.path.join("./data/cube/train/", image_src)))[
                :, :, :3
            ]
            .transpose(0, 1)
            .flip([0])
            .float()
            / 255.0
        )

        pixels /= torch.max(pixels)
        images.append(pixels)

    near = 0.5
    far = 7

    loss_fn = lambda outputs, labels: ((outputs - labels) ** 2).sum()
    model = NerfModel(scale)

    optimizer = torch.optim.Adam(model.parameters(), 0.0005)

    num_steps = 100000
    num_steps_to_render = 1000
    for step in range(num_steps):
        if step % num_steps_to_render == 0:
            print(f"rendering snapshot at step {step}")
            size = 100
            novel_view_transformation_matrix = generate_random_gimbal_transformation_matrix(1.0/scale)
            depth_image, color_image = render_image(size, novel_view_transformation_matrix, fov, near, far, model)

            out_depth_image = (depth_image.cpu().detach()).reshape((size, size)).t().fliplr().numpy()
            out_color_image = (color_image.cpu().detach() * 255).reshape((size, size, 3)).transpose(0, 1).flip([1]).numpy()
            imageio.imwrite(
                out_dir + f"depth_step_{step}.png",
                out_depth_image
            )
            imageio.imwrite(
                out_dir + f"color_step_{step}.png",
                out_color_image,
            )
            print("saved snapshot")
            #wandb_log({"random_gimbal_view_color": wandb.Image(out_color_image, mode="RGB"), "random_gimbal_view_depth": wandb.Image(out_depth_image)})
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
        )

        loss = loss_fn(colors.flatten(), expected_colors.flatten())
        loss.backward()

        optimizer.step()
        loss_at_step = loss.item()
        print(f"loss at step {step}: {loss_at_step}")
        wandb_log({"loss": loss_at_step, "step": step})


if __name__ == "__main__":
    train()
