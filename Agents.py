import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import util
import numpy as np

# Define model
class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.batch_size, self.channel, self.height, self.width = self.input_dim
        self.output_dim = output_dim
        # print(f"input_dim: {input_dim}, output_dim: {output_dim}")
        self.device = self.device()
        self.network = self.construct_network()
        self.network = self.network.to(self.device)
        print(self.network)

    def device(self):
        """ Retrive device for tensorflow"""
        device = ("cuda"if torch.cuda.is_available() else "mps"if torch.backends.mps.is_available() else "cpu")
        # print(f"Using {device} device")
        return device
    
    def construct_network(self):
        """
        Just to test for right now, the following network is from https://medium.com/@joachimiak.krzysztof/learning-to-play-pong-with-pytorch-tianshou-a9b8d2f1b8bd
        """
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
        # print(f"conv_out_size: {conv_out_size}")

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

    def forward(self, state):
        Q = self.network(state)
        # assert Q.requires_grad, "Q-Values must be a Torch Tensor with a Gradient"
        return Q


class ReplayMemory():
    """
    Different replay buffers in PyTorch
    https://pytorch.org/rl/tutorials/rb_tutorial.html
    """
    def __init__(self, capacity, batch_size, device):
        self.device = device
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(capacity, device=torch.device(device)))
        self.batch_size = batch_size
    def cache(self, state, action, reward, next_state, terminate):
        """
        Add (s, a, r, s') to memory
        """
        data = TensorDict( {
            "state": torch.Tensor(state),
            "action": torch.Tensor([action]),
            "reward": torch.Tensor([reward]),
            "next_state": torch.Tensor(next_state),
            "terminate": torch.Tensor([terminate])
        }, batch_size=[])
        self.memory.add(data)

    def recall (self):
        """
        Retrieve experience from memory
        """
        samples = self.memory.sample(self.batch_size).to(self.device)
        state, action, reward, next_state, terminate = (samples.get(key) for key in ("state", "action", "reward", "next_state", "terminate"))
        state = torch.tensor(state, device=self.device, dtype=torch.float) 
        action =  torch.tensor(action, device=self.device, dtype=torch.long) 
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float)
        terminate = torch.tensor(terminate, device=self.device, dtype=torch.float)
        return state, action.squeeze(), reward.squeeze(), next_state, terminate.squeeze()

    def __len__(self):
        return len(self.memory)

class DQN():
    def __init__(self, env, state_shape, num_actions, epsilon, alpha, gamma, sync_interval):
        self.env = env


        # The Neural Networks for The main Q network and the target network
        self.network = NeuralNet(state_shape, num_actions)
        self.target_net = NeuralNet(state_shape, num_actions)
        # Copy inital weights from Q Network into the target network
        self.target_net.load_state_dict(self.network.state_dict())
        # # Q_target parameters are frozen.
        # for p in self.target_net.parameters():
        #     p.requires_grad = False
        # Setup memory for DQN algorithm
        self.memory = ReplayMemory(10**4, 16, self.network.device)
        self.mem_batch_size = self.memory.batch_size

        self.step = 0 # Current Step of the agent

        # Hyperparameters
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.optimizer = optim.AdamW(self.network.parameters(), lr=alpha)
        self.loss_func = nn.SmoothL1Loss()
        self.sync_interval = sync_interval
        

    def action(self, state):
        # if epsilonn=0 then flipCoin returns False, if epsilon=1 then flipCoin returns True
        if util.flipCoin(self.epsilon):
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(state, device=self.network.device, dtype=torch.float32)
            q_vals_actions = self.network(state)
            action = torch.argmax(q_vals_actions).item()
        self.step += 1
        return action
    
    def current_q_w_estimate(self, state, action):
        current_Q = self.network(state)[np.arange(0, self.mem_batch_size), action]  
        return current_Q

    @torch.no_grad() # No Backwards computations needed
    def q_target(self, reward, next_state, terminate):
        target_Qs = self.network(next_state)
        best_action = torch.argmax(target_Qs).item()
        next_Q = self.target_net(next_state)[
            torch.arange(0, self.mem_batch_size), best_action
        ]
        not_done = 1 - terminate # Invert for mult below
        return (reward + self.gamma * next_Q*not_done).float()

    def update_network(self,q_w_estimate, q_target):
        loss = self.loss_func(q_w_estimate, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_w_to_target_net(self):
        self.target_net.load_state_dict(self.network.state_dict())
    
    def train(self):
        """
        Model learning/optimization
        """
        if self.step % self.sync_interval == 0:
            self.sync_w_to_target_net()
        if self.step % 10000 == 0: # Save every 1000 eps
            self.save_model()
        if self.step < 1000:
            return None, None
        state, action, reward, next_state, terminate = self.memory.recall()
        q_est = self.current_q_w_estimate(state, action)
        q_tgt = self.q_target(reward, next_state, terminate)
        loss = self.update_network(q_est, q_tgt)
        
        return (q_est.mean().item(), loss)

    def save_model(self):
        torch.save(dict(self.network.state_dict()),(f"./DQN/model_{int(self.step // self.sync_interval)}"))
        print(f'DQN Model saved at step: {self.step}')

    def load_model(self, model_path):
        self.network.load_state_dict(torch.load((model_path)))