import torch
import torch.nn as nn
import numpy as np
from abc import ABC,abstractmethod

class BaseNeuralNet(nn.Module,ABC):
    """
    A neural network base class for different types of neural networks used.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.channel, self.height, self.width = self.input_dim
        self.output_dim = output_dim
        self.device = self.device()
        # print(f"input_dim: {input_dim}, output_dim: {output_dim}")

        self.network = self.construct_network()
        self.network = self.network.to(self.device)
        # print(self.network)

    def device(self):
        """Retrieve device for tensorflow"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        # print(f'Using {device} device')
        return device
    
    @abstractmethod
    def construct_network(self) -> nn.Sequential:
        pass
    
    def get_conv_out_size(self, conv:nn.Sequential, image_dim:tuple[int,int,int,int]) -> int:
        """
        Calculate the size of the networks output based on its configuration and input by
        running on dummy numbers
        """
        return np.prod(conv(torch.rand(*image_dim)).shape)
    
    @abstractmethod
    def forward(self, state:np.ndarray) -> torch.Tensor:
        pass

class PixelNeuralNet(BaseNeuralNet):
    """
    A deep Neural Network designed with the following layers:
    Conv2d(3,32,8,4)
    ReLU()
    Conv2d(32,64,4,2)
    ReLU()
    Conv2d(64,64,3,1)
    ReLU()
    Flatten()
    Linear(1792,512)
    ReLU()
    Linear(512,10)
    """
    
    def construct_network(self) -> nn.Sequential:
        # Convolutional Layers
        conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self.get_conv_out_size(conv, self.input_dim)

        # Followed by Fully Connected Layers
        fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )
        
        return nn.Sequential(conv,fc)

    def forward(self, state:np.ndarray) -> torch.Tensor:
        Q : torch.Tensor = self.network(state)
        Q = Q.clone().detach().requires_grad_(True)
        assert Q.requires_grad, 'Q-values must be a torch Tensor with a gradient'
        return Q

class StateNeuralNet(BaseNeuralNet):
    """
    A deep Neural Network designed with the following layers:
    Linear(self.64, 256),
    ReLU()
    Linear(256, 128),
    ReLU()
    Linear(128, 128),
    ReLU()
    Linear(128, 128),
    ReLU()
    Linear(128, 64)
    """
    
    def construct_network(self) -> nn.Sequential:
        # Followed by Fully Connected Layers
        fc = nn.Sequential(
            nn.Linear(self.height * self.width, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )
        return fc
    
    def forward(self, state:np.ndarray) -> torch.Tensor:
        Q : torch.Tensor = self.network(state.reshape((-1,64)))
        Q = Q.clone().detach().requires_grad_(True)
        assert Q.requires_grad, 'Q-values must be a torch Tensor with a gradient'
        return Q