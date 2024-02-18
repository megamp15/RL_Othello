import torch
import torch.nn as nn
import util

# Define model
class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.channel, self.height, self.width = self.input_dim
        self.output_dim = output_dim
        print(f"input_dim: {input_dim}, output_dim: {output_dim}")
        self.device = self.device()

    def device(self):
        """ Retrive device for tensorflow"""
        device = ("cuda"if torch.cuda.is_available() else "mps"if torch.backends.mps.is_available() else "cpu")
        print(f"Using {device} device")
        return device
    
    def construct_network(self):
        """
        JUST TO TEST - COPY OF THE MARIO MINI CNN STRUCTURE FROM https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
        """
        print("Constructing network")
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

class ReplayMemory():
    """
    Different replay buffers in PyTorch
    https://pytorch.org/rl/tutorials/rb_tutorial.html
    """
    def __init__(self, capacity):
        pass

    def cache(self):
        """
        Add (s, a, r, s') to memory
        """
        pass

    def recall (self):
        """
        Retrieve experience from memory
        """
        pass


class DQN(NeuralNet):
    def __init__(self, env, state_shape, num_actions, epsilon):
        super().__init__(state_shape, num_actions)
        self.epsilon = epsilon
        self.env = env
        self.step = 0
        self.model = self.model()

    def model(self):
        network = self.construct_network()
        model = network.to(self.device)
        # print(model)
        return model

    def loss(self):
        """Custom Loss Function"""
        pass

    def train(self):
        """
        Model learning/optimization
        """
        pass

    def action(self, state):
        if util.flipCoin(self.epsilon):
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(state, device=self.device)
            q_vals_actions = self.model(state)
            action = torch.argmax(q_vals_actions, axis=1).item()
        self.step += 1
        return action
