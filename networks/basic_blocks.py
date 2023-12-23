import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size, activation):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        nn.init.xavier_uniform_(self.layers[-1].weight)
        for i in range(num_layers-1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            nn.init.xavier_uniform_(self.layers[-1].weight)
        self.layers.append(nn.Linear(hidden_size, output_size))
        nn.init.xavier_uniform_(self.layers[-1].weight)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)