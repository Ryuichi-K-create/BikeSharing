import torch
import torch.nn as nn

class BikeSharingMLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(50,), activation='relu'):
        super(BikeSharingMLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            in_dim = hidden_dim
            
        # Output layer (Regression)
        layers.append(nn.Linear(in_dim, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
