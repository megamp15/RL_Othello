from typing import TypedDict
from typing_extensions import Unpack, Required, NotRequired
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from enum import Enum
import random
from abc import ABC, abstractmethod
import os

from mem import ReplayMemory
from neuralNet import BaseNeuralNet
from othelloUtil import *


class AgentType(Enum):
    DQN = "DQN"
    DDQN = "DDQN"
    DUELDQN = "DUELDQN"
    SARSA = "SARSA"
    DSARSA = "DSARSA"
    DUELSARSA = "DUELSARSA"

class AgentParams(TypedDict):
    net_type : type[BaseNeuralNet]
    state_shape : tuple[int,int,int,int]
    num_actions : int
    epsilon : float
    epsilon_decay_rate : float
    epsilon_min : float
    alpha : float
    gamma : float
    sync_interval : int
    skip_training : int
    save_interval : int
    max_memory : int
    save_path : NotRequired[str]
    batch_size : int

class DeepAgent(ABC):
    def __init__(self, agent_type:AgentType, net_type:type[BaseNeuralNet], loss_func=nn.MSELoss, **kwargs : Unpack[AgentParams]) -> None:
        # The Neural Networks for The main Q network and the target network
        self.net_type = net_type
        self.network : BaseNeuralNet = self.net_type(kwargs['state_shape'], kwargs['num_actions'])
        self.num_actions = kwargs['num_actions']
        self.max_memory = kwargs['max_memory']

        # Save directory for model files
        self.save_path = kwargs['save_path']

        # Setup memory for DQN algorithm
        self.memory = ReplayMemory(self.max_memory, kwargs['batch_size'], self.network.device)
        self.mem_batch_size = self.memory.batch_size

        # Current Step of the agent
        self.step = 0

        # Hyperparameters
        self.epsilon = kwargs['epsilon']
        self.epsilon_decay_rate = kwargs['epsilon_decay_rate']
        self.epsilon_min = kwargs['epsilon_min']
        self.gamma = kwargs['gamma']
        
        self.skip_training = kwargs['skip_training'] # Skip the first n amount of training steps to cache experience in memory
        self.save_interval = kwargs['save_interval'] # Save the model every n steps

        self.optimizer = optim.AdamW(self.network.parameters(), lr=kwargs['alpha'], amsgrad=True)
        self.loss_func = loss_func()
        self.sync_interval = kwargs['sync_interval']
        self.agent_type = agent_type
        
    def set_state_shape(self,state_shape):
        self.state_shape= state_shape
    
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)

    def get_action_by_policy_(self, state:np.ndarray, available_moves:list[int], network:BaseNeuralNet) -> int:
        state = np.expand_dims(state, axis=0)
        state_t = torch.tensor(state, device=network.device, dtype=torch.float32)
        q_vals_actions = network(state_t)
        return self.clamp_illegal_actions(q_vals_actions,available_moves)
    
    def get_action_by_policy(self, state:np.ndarray, available_moves:list[int]) -> int:
        return self.get_action_by_policy_(state, available_moves, self.network)

    def get_action(self, state:np.ndarray, available_moves:list) -> tuple[int,int]:
        if available_moves == None or (isinstance(available_moves,list) and len(available_moves) == 0):
            print('No available moves.')
            return None
        if random.random() < self.epsilon:
            action = random.choice(available_moves)
        else:
            action = self.get_action_by_policy(state, available_moves)

        self.decay_epsilon()
        # self.step += 1

        return action
    
    def clamp_illegal_actions(self,q_vals_actions:torch.Tensor,available_moves:list)->int|None:
        if len(available_moves) == 0:
            return None
        q_vals = q_vals_actions.tolist()[0]
        available_q_vals = [q_vals[m] for m in available_moves]
        max_q_idx = available_q_vals.index(max(available_q_vals))
        max_q_move = available_moves[max_q_idx]
        return max_q_move
    
    def update(self, state:np.ndarray, action:int, reward:int, next_state:np.ndarray, next_action:int,term:bool=False) -> tuple:
        """
        Updates the Q values based on the next observed state
        """       
        self.memory.cache(state[0], action, reward, next_state[0], next_action, term)
        q_vals, loss = self.train()
        return q_vals, loss
    
    @abstractmethod
    def train(self, state, action, reward, next_state, next_action, terminate) -> tuple:
        self.memory.cache(state, action, reward, next_state, next_action, terminate)
        if self.step < self.skip_training:
            return None, None
    
    def get_Q_value_batch(self, network:BaseNeuralNet, state:np.ndarray, action_batch:list[int]=None) -> torch.Tensor:
        q_value_batch : torch.Tensor = network(state)
        if action_batch == None:
            action_batch = self.get_best_action_batch(q_value_batch)
        return q_value_batch[torch.arange(0,self.mem_batch_size), action_batch]
    
    def get_best_action_batch(self, q_value_batch:torch.Tensor) -> torch.Tensor:
        return torch.argmax(q_value_batch,dim=1).tolist()
    
    def current_q_w_estimate(self, state:np.ndarray, action:torch.Tensor) -> torch.Tensor:
        pred = self.network(state)
        current_Q = pred[np.arange(0, self.mem_batch_size), action]  
        return current_Q
    
    def reset(self) -> None:
        if self.step % self.save_interval == 0: # Save every n eps
            self.save_model()
        self.step += 1

    def update_network(self, q_w_estimate:torch.Tensor, q_target:torch.Tensor) -> float:
        loss : torch.Tensor = self.loss_func(q_w_estimate, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_model(self, save_path:str=None) -> None:
        if save_path == None:
            save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)
        file_name = f'{self.agent_type.value}_model_{int(self.step // self.save_interval)}'
        torch.save(
            dict(model=self.network.state_dict(), 
                 epsilon=self.epsilon
            ),
            f'{save_path}/{file_name}'
        )
        # print(f'\nDQN Model saved at step: {self.step}')

    def load_model(self, model_path:str) -> None:
        data : AgentParams = torch.load(model_path, map_location=self.network.device)
        self.network.load_state_dict(data.get('model'))
        self.epsilon = data.get('epsilon')
        print(f"Loading model at {model_path} with exploration rate {self.epsilon}")