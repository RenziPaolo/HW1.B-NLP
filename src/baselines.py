import torch.nn as nn
import torch

class RandomBase(nn.module):

    def __init__(self):
        super(RandomBase, self).__init__()
    
    def forward(self, x: list[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return torch.rand([128, 2])
