import torch.nn as nn
import torch.nn.functional as F

from mspEN.modules.types import DiscrType

class PatchGan(DiscrType):
    def __init__(self):
        super(PatchGan, self).__init__()

        self.main = nn.Sequential(
            # layer 1, halving dimensions
            nn.Conv2d(1, 64, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            # Layer 2
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            # Layer 3
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            
            # Layer 4 (final)
            nn.Conv2d(256, 1, 4, 2, 1, bias=False), 
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(output.shape[0], -1)

    def loss(self, pred, target, train_discriminator=True):
        b_size = pred.shape[0]
        pred = pred if train_discriminator else pred.clamp(0, 1)
        L = F.mse_loss(pred.view(-1), target.view(-1), reduction="none")
        L = L.view(b_size, -1).sum(dim=1)
        return L * 4
    