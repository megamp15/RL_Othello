import torch
import torch.nn as nn

# Define model
class NeuralNet(nn.Module):
    def __init__(self, state_dim, num_actions, epsilon):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.epsilon = epsilon
        print(f"state_dim: {state_dim}, num_actions: {num_actions}, epsilon: {epsilon}")
        self.device = self.device()
        self.network = self.network()

    def device(self):
        """ Retrive device for tensorflow"""
        device = ("cuda"if torch.cuda.is_available() else "mps"if torch.backends.mps.is_available() else "cpu")
        print(f"Using {device} device")
        return device
    
    def network(self):
        return nn.Sequential(
            # nn.Conv2d(in_channels=0, out_channels=32, kernel_size=8, stride=4),
            # nn.ReLU(),
            # nn.linear(),
            # nn.ReLU(),
            # nn.linear(128,num_actions),
        )

    def forward(self, state):
        Q = self.network(state)
        assert Q.requires_grad, "Q-Values must be a Torch Tensor with a Gradient"
        return Q
    
    def loss():
        pass

    def train():
        pass

    def action():
        pass


class DQN(NeuralNet):
    def __init__(self, state_shape, num_actions, epsilon):
        super().__init__(state_shape, num_actions, epsilon)

    def model(self):
        model = NeuralNet().to(self.device)
        print(model)