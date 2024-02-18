import torch
import torch.nn as nn

# Define model
class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.channel, self.height, self.width = self.input_dim
        self.output_dim = output_dim
        print(f"input_dim: {input_dim}, output_dim: {output_dim}")
        self.device = self.device()
        self.network = self.construct_network()

    def device(self):
        """ Retrive device for tensorflow"""
        device = ("cuda"if torch.cuda.is_available() else "mps"if torch.backends.mps.is_available() else "cpu")
        print(f"Using {device} device")
        return device
    
    def construct_network(self):
        """
        JUST TO TEST - COPY OF THE MARIO MINI CNN STRUCTURE FROM https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim),
        )

    def forward(self, state):
        Q = self.network(state)
        assert Q.requires_grad, "Q-Values must be a Torch Tensor with a Gradient"
        return Q
    
    


class DQN(NeuralNet):
    def __init__(self, state_shape, num_actions, epsilon):
        super().__init__(state_shape, num_actions, epsilon)

    def model(self):
        model = NeuralNet().to(self.device)
        print(model)

    def loss():
        pass

    def train():
        pass

    def action():
        pass