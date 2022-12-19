import torch

"""
points: tensor dim = (num points, space_dim)
"""


def embed_tensor(points, l):
    embeddings = points.repeat(1, l)
    scalers = torch.pow(2, torch.arange(0, l, 1)).repeat_interleave(3)

    # TODO unknown if this different ordering will be a problem with learning
    sin_embeddings = torch.sin(embeddings * scalers[None, :] * torch.pi)
    cos_embeddings = torch.cos(embeddings * scalers[None, :] * torch.pi)
    return torch.concat([sin_embeddings, cos_embeddings], dim=1)
