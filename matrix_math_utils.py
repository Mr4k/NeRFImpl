import torch


def generate_rot_x(angle):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, torch.cos(angle), -torch.sin(angle), 0],
            [0, torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    ).t()


def generate_rot_z(angle):
    return torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0, 0],
            [torch.sin(angle), torch.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).t()


def generate_translation(pos):
    return torch.tensor(
        [[1, 0, 0, pos[0]], [0, 1, 0, pos[1]], [0, 0, 1, pos[2]], [0, 0, 0, 1]]
    ).t()

def generate_random_gimbal_transformation_matrix(scale):
    return torch.matmul(torch.matmul(generate_translation(torch.tensor([0.0, 0, scale])), generate_rot_x(torch.pi * torch.rand(1))), generate_rot_z(torch.pi * torch.rand(1)))
