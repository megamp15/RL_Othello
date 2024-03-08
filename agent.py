from typing import TypedDict
from typing_extensions import Unpack, Required, NotRequired
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import gymnasium as gym
import os
from enum import Enum
import random
from abc import ABC, abstractmethod

from mem import ReplayMemory
from nn import NeuralNet

class AgentType(Enum):
    DQN = "DQN"
    DDQN = "DDQN"
    DUELDQN = "DUELDQN"
    SARSA = "SARSA"
    DSARSA = "DSARSA"
    DUELSARSA = "DUELSARSA"

class AgentParams(TypedDict):
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

class DeepAgent(ABC):
    def __init__(self, agent_type:AgentType, loss_func=nn.MSELoss, **kwargs : Unpack[AgentParams]) -> None:
        # The Neural Networks for The main Q network and the target network
        self.network = NeuralNet(kwargs['state_shape'], kwargs['num_actions'])
        self.num_actions = kwargs['num_actions']
        self.max_memory = kwargs['max_memory']

        # Save directory for model files
        self.save_path = kwargs['save_path']

        # Setup memory for DQN algorithm
        self.memory = ReplayMemory(self.max_memory, 32, self.network.device)
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

        self.optimizer = optim.AdamW(self.network.parameters(), lr=kwargs['alpha'])
        self.loss_func = loss_func
        self.sync_interval = kwargs['sync_interval']
        self.agent_type = agent_type
    
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)

    def get_action(self, state:np.ndarray, available_moves:list=None) -> int:
        # if epsilon=0 then flipCoin returns False, if epsilon=1 then flipCoin returns True
        if random.random() < self.epsilon:
            if available_moves != None and isinstance(available_moves,list):
                action = random.choice(available_moves)
            else:
                action = random.randint(0,self.num_actions-1)
        else:
            state = torch.tensor(state, device=self.network.device, dtype=torch.float32)
            q_vals_actions = self.network(state)
            action = torch.argmax(q_vals_actions).item()

        self.decay_epsilon()
        self.step += 1

        return action
    
    def update(self, state:np.ndarray, action:int, reward:int, next_state:np.ndarray, exit:bool=False) -> tuple:
        """
        Updates the Q values based on the next observed state
        """
        a_exit = False
        if len(self.memory) > self.max_memory:
            a_exit = True
        self.memory.cache(state[0], action, reward, next_state[0], exit)
        q_vals, loss = self.train()
        if len(self.memory) > self.max_memory:
            a_exit = True
        return q_vals, loss, a_exit
    
    @abstractmethod
    def train(self, save_path:str) -> tuple:
        pass
    
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
        file_name = f'{self.agent_type.value}_model_{int(self.step // self.save_interval)}'
        torch.save(dict(self.network.state_dict()),f'{self.save_path}/{file_name}')
        # print(f'\nDQN Model saved at step: {self.step}')

    def load_model(self, model_path:str) -> None:
        data = torch.load(model_path, map_location=self.network.device)
        self.network.load_state_dict(data.get('model'))
        self.epsilon = data.get('epsilon')
        print(f"Loading model at {model_path} with exploration rate {self.epsilon}")