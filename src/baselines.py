import torch.nn as nn
import torch

class RandomBase(nn.Module):

    def __init__(self, device):
        super(RandomBase, self).__init__()
        self.device = device
    
    def forward(self, x: list[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return torch.rand([x[0].size(0), 2], device=self.device)
