import torch

class NerfModel(torch.nn.Module):
    def __init__(self, scale, device):
        super(NerfModel, self).__init__()
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

        self.linear1 = torch.nn.Linear(pos_input_dims, 256)
        self.linear2 = torch.nn.Linear(256, 256)
        self.linear3 = torch.nn.Linear(256, 256)
        self.linear4 = torch.nn.Linear(256, 256)
        self.linear5 = torch.nn.Linear(256, 256)
        self.linear6 = torch.nn.Linear(256 + pos_input_dims, 256)
        self.linear7 = torch.nn.Linear(256, 256)
        self.linear8 = torch.nn.Linear(256, 256)
        self.linear9 = torch.nn.Linear(256, 256 + 1)
        self.linear10 = torch.nn.Linear(256 + dir_input_dims, 128)
        self.linear11 = torch.nn.Linear(128, 3)

        self.relu_activation = torch.nn.ReLU()
        self.sigmoid_activation = torch.nn.Sigmoid()

    def embed_tensor(self, points, l):
        embeddings = points.repeat(1, l)
        scalars = torch.pow(
                2.0, torch.arange(0, l, 1, device=self.device)
            ).repeat_interleave(3)

        # TODO unknown if this different ordering will be a problem with learning
        sin_embeddings = torch.sin(embeddings * scalars[None, :] * torch.pi)
        cos_embeddings = torch.cos(embeddings * scalars[None, :] * torch.pi)
        return torch.concat([sin_embeddings, cos_embeddings], dim=1)

    def forward(self, pos_input, dir_input):
        pos_embedding = self.embed_tensor(pos_input / self.scale, self.l_pos)

        x = self.linear1(pos_embedding)
        x = self.relu_activation(x)
        x = self.linear2(x)
        x = self.relu_activation(x)
        x = self.linear3(x)
        x = self.relu_activation(x)
        x = self.linear4(x)
        x = self.relu_activation(x)
        x = self.linear5(x)
        x = self.relu_activation(x)
        x = self.linear6(torch.concat([x, pos_embedding], dim=1))
        x = self.relu_activation(x)
        x = self.linear7(x)
        x = self.relu_activation(x)
        x = self.linear8(x)
        x = self.relu_activation(x)
        x = self.linear9(x)
        density = self.relu_activation(x[:, 0])
        dir_input = self.embed_tensor(dir_input, self.l_dir)
        x = self.linear10(torch.concat([x[:, 1:], dir_input], dim=1))
        x = self.linear11(x)
        color = self.sigmoid_activation(x)
        return color, density
