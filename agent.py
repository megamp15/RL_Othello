import torch
import torch.nn as nn
import torch.optim as optim
import util
import numpy as np
import time
import gymnasium as gym
import os

from mem import ReplayMemory
from nn import NeuralNet

class DeepAgent():
    def __init__(self, agent_type:str, env:gym.Env, state_shape:np.ndarray, num_actions:int, epsilon:float, epsilon_decay_rate:float, epsilon_min:float, alpha:float, gamma:float, sync_interval:int, skip_training:int, save_interval:int, loss_func = nn.MSELoss):
        self.env = env

        # The Neural Networks for The main Q network and the target network
        self.network = NeuralNet(state_shape, num_actions)

        # Setup memory for DQN algorithm
        self.memory = ReplayMemory(10**4, 16, self.network.device)
        self.mem_batch_size = self.memory.batch_size

        self.step = 0 # Current Step of the agent
        self.skip_training = skip_training # Skip the first n amount of training steps
        self.save_interval = save_interval # Save the model every n steps

        # Hyperparameters
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        self.optimizer = optim.AdamW(self.network.parameters(), lr=alpha)
        self.loss_func = loss_func()
        self.sync_interval = sync_interval
        self.agent_type = agent_type
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)

    def get_action(self, state:np.ndarray) -> int:
        self.decay_epsilon()
        # if epsilon=0 then flipCoin returns False, if epsilon=1 then flipCoin returns True
        if util.flipCoin(self.epsilon):
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(state, device=self.network.device, dtype=torch.float32)
            q_vals_actions = self.network(state)
            action = torch.argmax(q_vals_actions).item()
        self.step += 1
        return action
    
    def current_q_w_estimate(self, state:np.ndarray, action:torch.Tensor) -> float:
        current_Q = self.network(state)[np.arange(0, self.mem_batch_size), action]  
        return current_Q

    def update_network(self,q_w_estimate:float, q_target:float) -> float:
        loss = self.loss_func(q_w_estimate, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_model(self) -> None:
        folder_name = f'./{self.agent_type}'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        file_name = f'{time.strftime("%Y%m%d-%H%M%S")}_model_{int(self.step // self.save_interval)}'
        torch.save(dict(self.network.state_dict()),f'{folder_name}/{file_name}')
        # print(f'\nDQN Model saved at step: {self.step}')

    def load_model(self, model_path:str) -> None:
        self.network.load_state_dict(torch.load((model_path)))
    
