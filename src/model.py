
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 ** 2, 14 ** 2),
            nn.BatchNorm1d(14 ** 2),
            nn.ReLU(),
            nn.Linear(14 ** 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x.view(-1, 28 ** 2))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 14 ** 2),
            nn.BatchNorm1d(14 ** 2),
            nn.ReLU(),
            nn.Linear(14 ** 2, 28 ** 2),
        )

    def forward(self, x):
        return self.layers(x).view(-1, 1, 28, 28)
