import torch.nn as nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    """
    A feedforward neural network for classification tasks.

    Parameters:
    - num_features (int): Number of input features.
    - num_classes (int): Number of output classes.
    - layers (list): List specifying the number of neurons in each hidden layer.

    Attributes:
    - layers (nn.ModuleList): A container for holding the linear layers of the neural network.

    Methods:
    - __init__(num_features, num_classes, layers): Initializes the SimpleClassifier.
    - forward(x): Defines the forward pass of the neural network.
    """
    
    def __init__(self, num_features, num_classes, layers):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(num_features, layers[0]))

        # Hidden layers
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i - 1], layers[i]))

        # Output layer
        self.layers.append(nn.Linear(layers[-1], num_classes))

    def forward(self, x):
        # Pass through each layer except for the last one
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # Last layer with softmax
        x = self.layers[-1](x)

        return x
