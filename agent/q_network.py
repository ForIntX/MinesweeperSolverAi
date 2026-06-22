import math
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    CNN + Dueling DQN.
    Giriş : (batch, H*W) normalize state vektörü  (-1=kapalı, 0..1=açık/8)
    Çıkış : (batch, H*W) Q-değerleri
    """

    def __init__(self, input_dim, output_dim, board_h=None, board_w=None):
        super().__init__()
        if board_h is None:
            side = int(round(math.sqrt(input_dim)))
            board_h = board_w = side
        self.H = board_h
        self.W = board_w

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        flat = 64 * board_h * board_w

        self.val = nn.Sequential(nn.Linear(flat, 256), nn.ReLU(), nn.Linear(256, 1))
        self.adv = nn.Sequential(nn.Linear(flat, 256), nn.ReLU(), nn.Linear(256, output_dim))

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        b = x.size(0)
        h = self.conv(x.view(b, 1, self.H, self.W)).view(b, -1)
        v, a = self.val(h), self.adv(h)
        return v + a - a.mean(1, keepdim=True)
