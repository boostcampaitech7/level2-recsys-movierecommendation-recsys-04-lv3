import torch.nn as nn
import torch.nn.functional as F


class CDAE(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_hidden_units: int,
        corruption_ratio: float,
    ) -> None:
        super(CDAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_hidden_units = num_hidden_units
        self.corruption_ratio = corruption_ratio

        self.drop = nn.Dropout(self.corruption_ratio)
        self.encoder = nn.Linear(num_items, num_hidden_units)
        self.user_embedding = nn.Embedding(num_users, num_hidden_units)
        self.decoder = nn.Linear(num_hidden_units, num_items)

    def forward(self, input, user):
        h = F.normalize(input)
        h = self.drop(h)

        h = self.encoder(h)
        h += self.user_embedding(user)

        encoder = F.relu(h)

        return self.decoder(encoder)
