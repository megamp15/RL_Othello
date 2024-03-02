import torch
import torch.nn as nn
import torch.optim as optim
import util
import numpy as np
import time
import gymnasium as gym

from mem import ReplayMemory
from nn import NeuralNet

class DeepAgent():
    def __init__(self, env:gym.Env, state_shape:np.ndarray, num_actions:int, epsilon:float, alpha:float, gamma:float, sync_interval:int,loss_func = nn.SmoothL1Loss):
        self.env = env

        # The Neural Networks for The main Q network and the target network
        self.network = NeuralNet(state_shape, num_actions)

        # Setup memory for DQN algorithm
        self.memory = ReplayMemory(10**4, 16, self.network.device)
        self.mem_batch_size = self.memory.batch_size

        self.step = 0 # Current Step of the agent

        # Hyperparameters
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.optimizer = optim.AdamW(self.network.parameters(), lr=alpha)
        self.loss_func = loss_func
        self.sync_interval = sync_interval
        

    def get_action(self, state:np.ndarray) -> int:
        # if epsilonn=0 then flipCoin returns False, if epsilon=1 then flipCoin returns True
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
        torch.save(dict(self.network.state_dict()),(f"./DQN/{time.strftime('%Y%m%d-%H%M%S')}_model_{int(self.step // self.sync_interval)}"))
        print(f'DQN Model saved at step: {self.step}')

    def load_model(self, model_path:str) -> None:
        self.network.load_state_dict(torch.load((model_path)))
    

class DQN(DeepAgent):
    
    @torch.no_grad() # No Backwards computations needed
    def q_target(self, reward:torch.Tensor, next_state:torch.Tensor, terminate:torch.Tensor) -> float:
        target_Qs = self.network(next_state)
        best_action = torch.argmax(target_Qs).item()
        next_Q = self.network(next_state)[
            torch.arange(0, self.mem_batch_size), best_action
        ]
        not_done = 1 - terminate # Invert for mult below
        return (reward + self.gamma * next_Q*not_done).float()
    
    def train(self) -> tuple:
        """
        Model learning/optimization
        """
        if self.step % 10000 == 0: # Save every n eps
            self.save_model()
        if self.step < 10000:
            return None, None
        state, action, reward, next_state, terminate = self.memory.recall()
        q_est = self.current_q_w_estimate(state, action)
        q_tgt = self.q_target(reward, next_state, terminate)
        loss = self.update_network(q_est, q_tgt)
        
        return (q_est.mean().item(), loss)

class DDQN(DeepAgent):
    
    def __init__(self, env:gym.Env, state_shape:np.ndarray, num_actions:int, epsilon:float, alpha:float, gamma:float, sync_interval:int,loss_func = nn.SmoothL1Loss):
        super().__init__(env, state_shape, num_actions, epsilon, alpha, gamma, sync_interval, loss_func)
        self.target_net = NeuralNet(state_shape, num_actions)
        # Copy inital weights from Q Network into the target network
        self.target_net.load_state_dict(self.network.state_dict())
        # # Q_target parameters are frozen.
        # for p in self.target_net.parameters():
        #     p.requires_grad = False
        
    @torch.no_grad() # No Backwards computations needed
    def q_target(self, reward:torch.Tensor, next_state:torch.Tensor, terminate:torch.Tensor) -> float:
        target_Qs = self.network(next_state)
        best_action = torch.argmax(target_Qs).item()
        next_Q = self.target_net(next_state)[
            torch.arange(0, self.mem_batch_size), best_action
        ]
        not_done = 1 - terminate # Invert for mult below
        return (reward + self.gamma * next_Q*not_done).float()
    
    def sync_w_to_target_net(self) -> None:
        self.target_net.load_state_dict(self.network.state_dict())
        
    def train(self) -> tuple:
        """
        Model learning/optimization
        """
        if self.step % self.sync_interval == 0:
            self.sync_w_to_target_net()
        if self.step % 10000 == 0: # Save every n eps
            self.save_model()
        if self.step < 10000:
            return None, None
        state, action, reward, next_state, terminate = self.memory.recall()
        q_est = self.current_q_w_estimate(state, action)
        q_tgt = self.q_target(reward, next_state, terminate)
        loss = self.update_network(q_est, q_tgt)
        
        return (q_est.mean().item(), loss)

class DuelDQN(DeepAgent):
    
    def __init__(self, env:gym.Env, state_shape:np.ndarray, num_actions:int, epsilon:float, alpha:float, gamma:float, sync_interval:int,loss_func = nn.SmoothL1Loss):
        super().__init__(env, state_shape, num_actions, epsilon, alpha, gamma, sync_interval,loss_func)
        
        self.value_net = NeuralNet(state_shape, 1)
        self.advantage_net = self.network
        
        
    def current_q_w_estimate(self, state:np.ndarray, action:torch.Tensor) -> float:
        value = self.value_net(state)[np.arange(0, self.mem_batch_size), 1]
        advantages = self.network(state)[np.arange(0, self.mem_batch_size),:]
        mean_advantage = advantages.mean(dim=1)
        current_Q = value + (advantages - mean_advantage)
        return current_Q
    
    def q_target(self, reward:torch.Tensor, next_state:torch.Tensor, terminate:torch.Tensor) -> float:
        target_advantages = self.advantage_net(next_state)
        next_Q = torch.max(target_advantages,dim=1)
        not_done = 1 - terminate # Invert for mult below
        return (reward + self.gamma * next_Q*not_done).float()

    
    def train(self) -> tuple:
        """
        Model learning/optimization
        """
        if self.step % 10000 == 0: # Save every n eps
            self.save_model()
        if self.step < 10000:
            return None, None
        state, action, reward, next_state, terminate = self.memory.recall()
        q_est = self.current_q_w_estimate(state, action)
        q_tgt = self.q_target(reward, next_state, terminate)
        loss = self.update_network(q_est, q_tgt)
        
        return (q_est.mean().item(), loss)
