import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import util
import numpy as np

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
        self.network = self.network.to(self.device)
        print(self.network)

    def device(self):
        """ Retrive device for tensorflow"""
        device = ("cuda"if torch.cuda.is_available() else "mps"if torch.backends.mps.is_available() else "cpu")
        print(f"Using {device} device")
        return device
    
    def construct_network(self):
        """
        Just to test for right now, the following network is from https://medium.com/@joachimiak.krzysztof/learning-to-play-pong-with-pytorch-tianshou-a9b8d2f1b8bd
        """
        print("Constructing network")
        # Convolutional Layers
        conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self.get_conv_out_size(conv, self.input_dim)
        print(f"conv_out_size: {conv_out_size}")

        # Followed by Fully Connected Layers
        fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )
        
        return torch.nn.Sequential(*(list(conv)+list(fc)))

    def get_conv_out_size(self, conv, image_dim):
        return np.prod(conv(torch.rand(*image_dim)).data.shape)

    # def forward(self, state):
    #     print("forward")
    #     Q = self.network(state)
    #     assert Q.requires_grad, "Q-Values must be a Torch Tensor with a Gradient"
    #     return Q

    def forward(self, state):
        # print("Input shape:", state.shape)
        
        # Pass through convolutional layers
        state = self.network[:6](state)
        # print("After convolutional layers shape:", state.shape)
        
        #TODO: # I'm not sure why but the nn.Flatten() causes an error between the conv layer and the FC layer
        # The error is in matrix mult of 64 x 28 to 1796 x 512. 1796 is the conv_out_size
        # Pass through flattening layer
        state = torch.flatten(state)      
        # print("After flattening shape:", state.shape)
        
        # Pass through first linear layer
        state = self.network[7](state)
        # print("After first linear layer shape:", state.shape)
        
        # Pass through ReLU activation
        state = self.network[8](state)
        
        # Pass through second linear layer
        state = self.network[9](state)
        # print("Output shape:", state.shape)
        
        return state

class ReplayMemory():
    """
    Different replay buffers in PyTorch
    https://pytorch.org/rl/tutorials/rb_tutorial.html
    """
    def __init__(self, capacity, device):
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(capacity, device=torch.device(device)))
        self.batch_size = 32
    def cache(self, state, action, reward, next_state):
        """
        Add (s, a, r, s') to memory
        """
        data = TensorDict( {
            "state": torch.Tensor(state),
            "action": torch.Tensor([action]),
            "reward": torch.Tensor([reward]),
            "next_state": torch.Tensor(next_state)
        }, batch_size=[])
        self.memory.add(data)

    def recall (self):
        """
        Retrieve experience from memory
        """
        samples = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward = (samples.get(key) for key in ("state", "next_state", "action", "reward"))
        return state, action.squeeze(), reward.squeeze(), next_state

    def __len__(self):
        return len(self.memory)

class DQN(NeuralNet):
    def __init__(self, env, state_shape, num_actions, epsilon, model_path=None):
        super().__init__(state_shape, num_actions)
        self.epsilon = epsilon
        self.env = env
        self.step = 0
        self.memory = ReplayMemory(100000, self.device)

    def train(self):
        """
        Model learning/optimization
        """
        
        

    def get_Q(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        q_vals = self.forward(state)
        return q_vals
    
    def action(self, state):
        if util.flipCoin(self.epsilon):
            action = self.env.action_space.sample()
        else:
            q_vals_actions = self.get_Q(state)
            # print(q_vals_actions)
            action = torch.argmax(q_vals_actions).item()
        self.step += 1
        return action