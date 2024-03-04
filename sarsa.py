import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from nn import NeuralNet

from agent import DeepAgent

class SARSA(DeepAgent):
    
    @torch.no_grad() # No Backwards computations needed
    def q_target(self, reward:torch.Tensor, next_state:torch.Tensor, next_action:torch.Tensor, terminate:torch.Tensor) -> float:
        target_Qs = self.network(next_state)
        next_Q = self.network(next_state)[
            torch.arange(0, self.mem_batch_size), next_action
        ]
        not_done = 1 - terminate # Invert for mult below
        return (reward + self.gamma * next_Q*not_done).float()
    
    def train(self) -> tuple:
        """
        Model learning/optimization
        """
        if self.step < self.skip_training:
            return None, None
        if self.step % self.save_interval == 0: # Save every n eps
            self.save_model()
        
        state, action, reward, next_state, next_action, terminate = self.memory.recall()
        q_est = self.current_q_w_estimate(state, action)
        q_tgt = self.q_target(reward, next_state, next_action, terminate)
        loss = self.update_network(q_est, q_tgt)
        
        return (q_est.mean().item(), loss)

class SARSA_DDQN(DeepAgent):
    
    def __init__(self, env:gym.Env, state_shape:np.ndarray, num_actions:int, epsilon:float, alpha:float, gamma:float, sync_interval:int, skip_training:int, save_interval:int, loss_func = nn.MSELoss):
        super().__init__(env, state_shape, num_actions, epsilon, alpha, gamma, sync_interval, skip_training, save_interval, loss_func)
        self.target_net = NeuralNet(state_shape, num_actions)
        # Copy inital weights from Q Network into the target network
        self.target_net.load_state_dict(self.network.state_dict())
        # # Q_target parameters are frozen.
        # for p in self.target_net.parameters():
        #     p.requires_grad = False
        
    @torch.no_grad() # No Backwards computations needed
    def q_target(self, reward:torch.Tensor, next_state:torch.Tensor, next_action:torch.Tensor, terminate:torch.Tensor) -> float:
        target_Qs = self.network(next_state)
        next_Q = self.target_net(next_state)[
            torch.arange(0, self.mem_batch_size), next_action
        ]
        not_done = 1 - terminate # Invert for mult below
        return (reward + self.gamma * next_Q*not_done).float()
    
    def sync_w_to_target_net(self) -> None:
        self.target_net.load_state_dict(self.network.state_dict())
        
    def train(self) -> tuple:
        """
        Model learning/optimization
        """
        if self.step < self.skip_training:
            return None, None
        if self.step % self.sync_interval == 0:
            self.sync_w_to_target_net()
        if self.step % self.save_interval == 0: # Save every n eps
            self.save_model()
        
        state, action, reward, next_state, next_action, terminate = self.memory.recall()
        q_est = self.current_q_w_estimate(state, action)
        q_tgt = self.q_target(reward, next_state, next_action, terminate)
        loss = self.update_network(q_est, q_tgt)
        
        return (q_est.mean().item(), loss)

class SARSA_DuelDQN(DeepAgent):
    
    def __init__(self, env:gym.Env, state_shape:np.ndarray, num_actions:int, epsilon:float, alpha:float, gamma:float, sync_interval:int, skip_training:int, save_interval:int, loss_func = nn.MSELoss):
        super().__init__(env, state_shape, num_actions, epsilon, alpha, gamma, sync_interval, skip_training, save_interval, loss_func)
        
        self.value_net = NeuralNet(state_shape, 1)
        self.advantage_net = self.network
        
        
    def current_q_w_estimate(self, state:np.ndarray, action:torch.Tensor) -> float:
        value = self.value_net(state)[np.arange(0, self.mem_batch_size), 0]
        advantages = self.network(state)[np.arange(0, self.mem_batch_size),:]
        mean_advantage = advantages.mean(dim=1)
        current_Q = value + (advantages[np.arange(0, self.mem_batch_size),action.tolist()] - mean_advantage)
        return current_Q
    
    def q_target(self, reward:torch.Tensor, next_state:torch.Tensor, next_action:torch.Tensor, terminate:torch.Tensor) -> float:
        target_advantages = self.advantage_net(next_state)
        next_Q = target_advantages[torch.arange(0, self.mem_batch_size), next_action]
        not_done = 1 - terminate # Invert for mult below
        return (reward + self.gamma * next_Q*not_done).float()

    
    def train(self) -> tuple:
        """
        Model learning/optimization
        """
        if self.step < self.skip_training:
            return None, None
        if self.step % self.save_interval == 0: # Save every n eps
            self.save_model()
        
        state, action, reward, next_state, next_action, terminate = self.memory.recall()
        q_est = self.current_q_w_estimate(state, action)
        q_tgt = self.q_target(reward, next_state, next_action, terminate)
        loss = self.update_network(q_est, q_tgt)
        
        return (q_est.mean().item(), loss)

