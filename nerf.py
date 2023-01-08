import json
import torch

import imageio.v3 as iio

import os

def inverse_transform_sampling(device, stopping_probs, bin_boundary_points, num_samples):
    """
    Allocates sample points over bins based on the relative probabiltiy the ray terminates inside that particular bin

    Args:
        device: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device 
        stopping_probs: Size (batch_size, num_bins) tensor representing the probability of a ray terminating at each bin
        bin_boundary_points: Size (batch_size, num_bins + 1) tensor representing the end of each bin
        num_samples: number of points to sample
    Returns:
        Size (batch_size, num_samples) tensor representing the samples along the ray in sorted order
    """
    batch_size, _ = stopping_probs.shape
    epsilon = 0.00001
    sample_bins = torch.multinomial(stopping_probs + epsilon, num_samples, True)
    indexes = torch.arange(0, batch_size, 1, device=device).repeat_interleave(num_samples)
    flatten_samples = sample_bins.flatten()
    return (
        (
            bin_boundary_points[indexes, flatten_samples + 1]
            - bin_boundary_points[indexes, flatten_samples]
        )
        * torch.rand((batch_size * num_samples), device=device)
        + bin_boundary_points[indexes, flatten_samples]
    ).view(batch_size, -1)


def compute_stratified_sample_points(device, batch_size, num_samples, t_near, t_far):
    """
    Computes sampling points in even space bins along a ray between a near and far distance along that ray
    Returns an arr of n sampling points

    Args:
        device: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device 
        batch_size: the size of the batch
        num_samples: int
        t_near: Size (batch_size) tensor representing the time along the ray representing the near plane
        t_far: Size (batch_size) tensor representing the time along the ray representing the far plane
    
    Returns:
        Size (batch_size, num_samples) tensor representing the samples along the ray in sorted order
    """
    bin_width = t_far - t_near
    return (
        torch.tensor(t_near).to(device).view(-1, 1).repeat(1, num_samples)
        + bin_width.view(-1, 1)
        * (
            torch.rand((batch_size, num_samples), device=device)
            + torch.arange(num_samples, device=device).repeat(batch_size, 1)
        )
        / num_samples
    )


def radiance_field_output(network, points, dirs):
    """
    Evaluates a radiance field represented by a neural network

    Args:
        network: the neural network representing the radiance field
        points: Size (batch_size, 3) tensor representing points to sample from the radiance field
        dirs: Size (batch_size, 3) tensor representing directions associated with each point
    Returns:
        a tuple (colors, opacity)
    """
    return network(points, dirs)


def trace_hierarchical_ray(
    device,
    coarse_radiance_field,
    fine_radiance_field,
    num_coarse_sample_points,
    num_extra_fine_sample_points,
    positions,
    directions,
    t_near,
    t_far,
    background_color,
):
    """
    Traces two different rays through the coarse and fine radiance fields.
    First traces a ray through the coarse field then samples point for the fine field based on inverse transform sampling.
    Finally we calculate the final color and deoth by sampling from the fine field

    Args:
        device: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
        coarse_radiance_field: 
        fine_radiance_field:
        num_coarse_sample_points: the number of points to sample from the coarse field
        num_extra_fine_sample_points: the number of extra points to sample from the fine field in addition to num_coarse_sample_points
        positions: Size (batch_size, 3) tensor representing the ray origins in 3d space
        directions: Size (batch_size, 3) tensor representing the normalized ray directions in 3d space
        t_near: Size (batch_size) tensor representing the time along the ray representing the near plane
        t_far: Size (batch_size) tensor representing the time along the ray representing the far plane
        background_color = Size (3) tensor representing the color at the far plane

    Returns:
        fine_color: Size (batch_size) tensor representing the colors along each fine ray
        fine_expected_distance: Size (batch_size) tensor representing the expected distance to collision along each fine ray
        coarse_color: Size (batch_size) tensor representing the colors along each coarse ray
    """
    coarse_radiance_field = coarse_radiance_field.to(device)
    fine_radiance_field = fine_radiance_field.to(device)
    positions = positions.to(device)
    directions = directions.to(device)
    t_near = t_near.to(device)
    t_far = t_far.to(device)
    background_color = background_color.to(device)

    batch_size = positions.shape[0]

    coarse_stratified_sample_times = compute_stratified_sample_points(
        device, batch_size, num_coarse_sample_points + 1, t_near, t_far
    )
    coarse_color, _, stopping_probs = trace_ray(
        device,
        coarse_stratified_sample_times,
        coarse_radiance_field,
        positions,
        directions,
        num_coarse_sample_points,
        t_near,
        t_far,
        background_color,
    )

    inverse_transform_sample_times = inverse_transform_sampling(
        device, stopping_probs, coarse_stratified_sample_times, num_extra_fine_sample_points
    )

    fine_sample_times, _ = torch.sort(
        torch.concat(
            [coarse_stratified_sample_times, inverse_transform_sample_times], dim=1
        ),
        dim=1,
    )

    fine_color, fine_distance, _ = trace_ray(
        device,
        fine_sample_times,
        fine_radiance_field,
        positions,
        directions,
        num_coarse_sample_points + num_extra_fine_sample_points,
        t_near,
        t_far,
        background_color,
    )

    return fine_color.cpu(), fine_distance.cpu(), coarse_color.cpu()


def trace_ray(
    device, stratified_sample_times, radiance_field, origins, directions, t_near, t_far, background_color
):
    """
    Traces a ray through a radiance field

    Args:
        device: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
        stratified_sample_times: Size (batch_size, num_samples) tensor representing the ordered times to sample along each ray
        positions: Size (batch_size, 3) tensor representing the ray origins in 3d space
        directions: Size (batch_size, 3) tensor representing the normalized ray directions in 3d space
        t_near: Size (batch_size) tensor representing the time along the ray representing the near plane
        t_far: Size (batch_size) tensor representing the time along the ray representing the far plane
        background_color = Size (3) tensor representing the color at the far plane

    Returns:
        color: Size (batch_size) tensor representing the colors along each ray
        expected_distance: Size (batch_size) tensor representing the expected distance to collision along each ray
        stopping_probs: Size (batch_size, num_samples) tensor representing the probabiltiy of the ray terminating in each bin
    """
    num_samples = stratified_sample_times.shape[1]
    batch_size = origins.shape[0]

    stratified_sample_points_centered_at_the_origin = stratified_sample_times.view(
        batch_size, num_samples + 1, 1
    ).repeat(1, 1, 3) * directions.view(batch_size, 1, -1).repeat(1, num_samples + 1, 1)
    stratified_sample_points = (
        stratified_sample_points_centered_at_the_origin
        + origins.view(batch_size, 1, -1).repeat(1, num_samples + 1, 1)
    )

    colors, opacity = radiance_field_output(
        radiance_field,
        stratified_sample_points.view(-1, 3),
        directions.repeat_interleave(num_samples + 1, dim=0),
    )
    colors = colors.view(batch_size, num_samples + 1, 3)
    opacity = opacity.view(batch_size, num_samples + 1)

    cum_partial_passthrough_sum = torch.zeros(batch_size, device=device)

    # a tensor that gives the probablity of the ray terminating in the nth bin
    # note this is really only need for the course network
    # might want to refactor the code so it can be disabled for the fine network
    stopping_probs = torch.zeros((batch_size, num_samples), device=device)
    cum_color = torch.zeros((batch_size, 3), device=device)
    cum_expected_distance = torch.zeros(batch_size, device=device)
    distance_acc = torch.ones(batch_size, device=device) * t_near

    # TODO (getting rid of this for loop likely speeds up rendering)
    # on second thought maybe not, bottleneck will eventually likely be get_network_output
    for i in range(num_samples):
        delta = stratified_sample_times[:, i + 1] - stratified_sample_times[:, i]
        prob_hit_current_bin = 1 - torch.exp(-opacity[:, i] * delta)
        cum_passthrough_prob = torch.exp(-cum_partial_passthrough_sum)

        stopping_probs[:, i] = cum_passthrough_prob * prob_hit_current_bin

        cum_color += stopping_probs[:, i].view(-1, 1).repeat(1, 3) * colors[:, i]

        cum_expected_distance += stopping_probs[:, i] * distance_acc

        cum_partial_passthrough_sum += opacity[:, i] * delta
        distance_acc += delta

    # add far plane
    cum_passthrough_prob = torch.exp(-cum_partial_passthrough_sum)
    cum_expected_distance += cum_passthrough_prob * t_far
    cum_color += cum_passthrough_prob.view(-1, 1).matmul(background_color.view(1, 3))

    return (
        cum_color,
        torch.min(
            torch.max(cum_expected_distance, torch.tensor(t_near)), torch.tensor(t_far)
        ),
        stopping_probs,
    )

def replace_alpha_with_solid_color(img, background_color):
    img = img / 255.0
    img_alpha_channel = img[:, :, 3][:, :, None]
    background_contribution = torch.matmul(1.0 - img_alpha_channel, background_color.view(1, -1))
    foreground_contribution = img_alpha_channel * img[:, :, :3]
    return background_contribution + foreground_contribution

def load_config_file(data_path, type, background_color):
    """
    Loads a json config file in the style of the original NeRF paper's training data

    Args:
        data_path: the path to the folder containing the config file
        type: the type of config to load from the file, "train", "val" or "test"
        background_color: Size (3) tensor representing the color to swap out the alpha channel with

    Returns:
        fov: the field of view in radians
        images: Size (num_cameras, width, height, 3) tensor containing the images associated with the views in this set
        transformation_matricies: Size(num_cameras, 4, 4) tensor containing the 4x4 transformation matricies that define the position of each camera
    """
    with open(os.path.join(data_path, f"transforms_{type}.json")) as f:
        config = json.load(f)

    if "frames" not in config:
        raise Exception(
            f"keyword 'frames' not found in config file at path: {data_path}"
        )

    if "camera_angle_x" not in config:
        raise Exception(
            f"keyword 'amera_angle_x' not found in config file at path: {data_path}"
        )

    transformation_matricies = []
    images = []
    fov = None
    if "camera_angle_x" in config:
        fov = torch.tensor(config["camera_angle_x"])
    for f in config["frames"]:
        transformation_matrix = torch.tensor(
            f["transform_matrix"], dtype=torch.float
        ).t()
        transformation_matricies.append(transformation_matrix)
        image_src = f["file_path"] + ".png"
        pixels = (
            replace_alpha_with_solid_color(torch.tensor(iio.imread(os.path.join(data_path, image_src))), background_color)
            .transpose(0, 1)
            .flip([0])
            .float()
        )
        #pixels /= torch.max(pixels)
        width, height, channels = pixels.shape

        assert width == height
        assert channels == 3

        images.append(pixels)

    transformation_matricies = torch.stack(transformation_matricies)
    images = torch.stack(images)
    return fov, images, transformation_matricies


def generate_rays(fov, camera_rotation_matrix, image_plane_points, aspect_ratio=1):
    """
    Generates rays for a camera whose rotatation matrix is represented by 

    Args:
        fov: angle of field of view in radians
        camera_transformation_matrix: a 3x3 matrix representing the rotation of the camera
        image_plane_points: (batch_size, 2), the locations of the pixels on the screen in [0, 1]^2
        aspect_ratio: width / height

    Returns:
        Size (batch_size, 3) tensor representing the rays corresponding to the points on the image plan
    """
    batch_size = image_plane_points.shape[0]
    x = (2 * image_plane_points[:, 0] - 1) * torch.tan(fov / 2.0) * aspect_ratio
    y = (2 * image_plane_points[:, 1] - 1) * torch.tan(fov / 2.0)
    z = torch.tensor([1.0]).repeat(batch_size)
    ray = torch.stack([x, y, z], dim=-1)
    ray = torch.mm(ray, camera_rotation_matrix)
    ray /= ray.norm(dim=1).view(-1, 1).repeat(1, 3)
    return -ray


def get_camera_position(camera_transformation_matrix):
    ray = torch.tensor([[0, 0, 0, 1.0]])
    return torch.mm(ray, camera_transformation_matrix)[0, :3]
