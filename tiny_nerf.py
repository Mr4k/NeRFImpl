import torch

class TinyNerfModel(torch.nn.Module):
    def __init__(self, scale, device):
        super(TinyNerfModel, self).__init__()
        self.device = device
        self.to(device)

        self.scale = scale
        self.l_pos = 10
        self.l_dir = 4

        pos_input_dims = self.l_pos * 2 * 3
        dir_input_dims = self.l_dir * 2 * 3

        self.scalers = {
            self.l_pos: torch.pow(
                2, torch.arange(0, self.l_pos, 1, device=device)
            ).repeat_interleave(3),
            self.l_dir: torch.pow(
                2, torch.arange(0, self.l_dir, 1, device=device)
            ).repeat_interleave(3),
        }

        self.linear1 = torch.nn.Linear(pos_input_dims, 128 + 1)
        self.linear2 = torch.nn.Linear(128 + dir_input_dims, 3)

        self.relu_activation = torch.nn.ReLU()
        self.sigmoid_activation = torch.nn.Sigmoid()

    def embed_tensor(self, points, l):
        embeddings = points.repeat(1, l)

        # TODO unknown if this different ordering will be a problem with learning
        sin_embeddings = torch.sin(embeddings * self.scalers[l][None, :] * torch.pi)
        cos_embeddings = torch.cos(embeddings * self.scalers[l][None, :] * torch.pi)
        return torch.concat([sin_embeddings, cos_embeddings], dim=1)

    def forward(self, pos_input, dir_input):
        pos_embedding = self.embed_tensor(pos_input / self.scale, self.l_pos)

        x = self.linear1(pos_embedding)
        x = self.relu_activation(x)
        density = self.relu_activation(x[:, 0])
        dir_input = self.embed_tensor(dir_input, self.l_dir)
        x = self.linear2(torch.concat([x[:, 1:], dir_input], dim=1))
        color = self.sigmoid_activation(x)
        return color, density
