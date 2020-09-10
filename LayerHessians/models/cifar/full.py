import torch
import torch.nn as nn

class FullyConnectedBlock(nn.Module):

    def __init__(self, hidden_dim, num_hidden=2, skip=False):

        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.skip = skip
        self.lin = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim))

    def forward(self, x):

        out = self.lin(x)
        if self.skip:
            out = out + x
        return out


class FullyConnected(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, num_hidden, skip=False):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.skip = skip
        
        self.layers = nn.Sequential()

        self.layers.add_module('init', nn.Linear(self.input_dim, self.hidden_dim))
        for i in range(self.num_hidden - 2):
            self.layers.add_module('hidden%d' % i, FullyConnectedBlock(self.hidden_dim, skip=self.skip))
        self.layers.add_module('last', nn.Linear(self.hidden_dim, self.num_classes))

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.layers(x)

        return x