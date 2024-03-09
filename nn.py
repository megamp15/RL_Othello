import torch
import torch.nn as nn
import numpy as np

class NeuralNet(nn.Module):
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
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.batch_size, self.channel, self.height, self.width = self.input_dim
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
    
    def construct_network(self) -> nn.Sequential:
        """
        Just to test for right now, the following network is from 
        
        https://medium.com/@joachimiak.krzysztof/learning-to-play-pong-with-pytorch-tianshou-a9b8d2f1b8bd
        """
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
        # print(f"conv_out_size: {conv_out_size}")

        # Followed by Fully Connected Layers
        fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )

        # conv = nn.Sequential(
        #     nn.Conv2d(in_channels=self.channel, out_channels=32, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        # )
        # conv_out_size = self.get_conv_out_size(self.conv, self.input_dim)

        # fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(self.conv_out_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, self.output_dim)
        # )
        
        return nn.Sequential(conv,fc)

    def get_conv_out_size(self, conv:nn.Sequential, image_dim:tuple[int,int,int,int]) -> int:
        """
        Calculate the size of the networks output based on its configuration and input by
        running on dummy numbers
        """
        return np.prod(conv(torch.rand(*image_dim)).shape)

    def forward(self, state:np.ndarray):
        Q = self.network(state)
        # assert Q.requires_grad, "Q-Values must be a Torch Tensor with a Gradient"
        return Q

