import uuid
import torch
#import wandb
from batch_and_sampler import render_image, render_rays, sample_batch
from matrix_math_utils import (
    generate_random_hemisphere_gimbal_transformation_matrix,
)
from nerf import load_config_file

from neural_nerf import NerfModel

import os

#from wandb_wrapper import wandb_init, wandb_log

from itertools import chain

import argparse

import imageio


def train(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        print("cuda acceleration available. Using cuda")

    # wandb_init({"entity": "mr4k", "project": "nerf"})

    training_run_id = uuid.uuid4()
    out_dir = f"./training_output/runs/{training_run_id}/"

    print(f"creating output dir: {out_dir}")
    os.makedirs(out_dir)

    background_color = torch.tensor([1.0, 1.0, 1.0])

    # (batch size, 3)
    scale = 5.0
    batch_size = 3500

    train_fov, train_images, train_transformation_matricies = load_config_file(os.path.join(args.data_path), "train", background_color)
    _, _, val_transformation_matricies = load_config_file(os.path.join(args.data_path), "val", background_color)

    near = 0.5
    far = 7

    loss_fn = lambda outputs, labels: ((outputs - labels) ** 2).sum()
    coarse_model = NerfModel(scale, device).to(device)
    fine_model = NerfModel(scale, device).to(device)

    optimizer = torch.optim.Adam(
        chain(coarse_model.parameters(), fine_model.parameters()), lr=0.0005
    )

    num_steps = 100000
    loss_at_last_snapshot = -1
    last_snapshot_iter = -1
    for step in range(num_steps):
        camera_poses, rays, distance_to_depth_modifiers, expected_colors = sample_batch(
            batch_size,
            200,
            train_transformation_matricies,
            train_images,
            train_fov,
        )

        optimizer.zero_grad()

        _, colors, coarse_colors = render_rays(
            batch_size,
            camera_poses,
            rays,
            distance_to_depth_modifiers,
            near,
            far,
            coarse_model,
            fine_model,
            device,
            background_color,
        )

        loss = loss_fn(colors.flatten(), expected_colors.flatten()) + loss_fn(
            coarse_colors.flatten(), expected_colors.flatten()
        )
        loss.backward()

        optimizer.step()
        loss_at_step = loss.item()
        print(f"loss at step {step}: {loss_at_step}")
        #wandb_log({"loss": loss_at_step, "step": step})

        take_snapshot_iter = (
            args.snapshot_iters >= 0 and (last_snapshot_iter < 0 or step - last_snapshot_iter >= args.snapshot_iters)
        )
        take_snapshot_loss = args.snapshot_train_loss_percentage >= 0 and (
            loss_at_last_snapshot < 0
            or loss < loss_at_last_snapshot * args.snapshot_train_loss_percentage
        )

        if take_snapshot_iter or take_snapshot_loss:
            loss_at_last_snapshot = loss
            last_snapshot_iter = step
            for i, transformation_matrix in enumerate(list(val_transformation_matricies[0:args.number_of_validation_views])):
                print(
                    f"rendering snapshot from view {i} at step {step} with loss {loss}"
                )
                size = 200

                with torch.no_grad():
                    depth_image, color_image, coarse_color_image = render_image(
                        size,
                        transformation_matrix,
                        train_fov,
                        near,
                        far,
                        coarse_model,
                        fine_model,
                        device,
                        background_color
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
                out_coarse_color_image = (
                    (coarse_color_image.cpu().detach() * 255)
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
                imageio.imwrite(
                    out_dir + f"view_{i}_coarse_color_step_{step}.png",
                    out_coarse_color_image,
                )
                print("saved snapshot")
                optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeRF")
    parser.add_argument(
        "--data_path",
        help="directiory containing the desired transforms_train.json file",
    )
    parser.add_argument(
        "--snapshot_iters",
        type=int,
        help="the max number of iterations since the last snapshot after which to take a snapshot. Negative means don't do snapshots based on number of iterations",
        default=-1,
    )
    parser.add_argument(
        "--snapshot_train_loss_percentage",
        type=float,
        help="if the loss is < percentage * last snapshot loss, take a snapshot. Negative means don't do snapshots based on this",
        default=-1,
    )
    parser.add_argument(
        "--number_of_validation_views",
        type=int,
        help="the number of validation views to render on snapshot",
        default=5,
    )
    args = parser.parse_args()
    train(args)
