import torch
import torch.nn as nn

class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLayer, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.linear(x)
        return x