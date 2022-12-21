import torch


def generate_rot_x(angle):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, torch.cos(angle), -torch.sin(angle), 0],
            [0, torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def generate_rot_z(angle):
    return torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0, 0],
            [torch.sin(angle), torch.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def generate_translation(pos):
    return torch.tensor(
        [[1, 0, 0, pos[0]], [0, 1, 0, pos[1]], [0, 0, 1, pos[2]], [0, 0, 0, 1]]
    )
