from typing_extensions import Unpack
import torch
import torch.nn as nn
import numpy as np
import random

from agent import DeepAgent, AgentType, AgentParams

class DQN(DeepAgent):
    """
    A deep q learning network agent
    """
    def __init__(self,**kwargs:Unpack[AgentParams]) -> None:
        super().__init__(agent_type=AgentType.DQN, **kwargs)

    @torch.no_grad() # No Backwards computations needed
    def q_target(self, reward:torch.Tensor, next_state:torch.Tensor, terminate:bool) -> torch.Tensor:
        target_Qs = self.get_Q_value_batch(self.network, next_state)

        next_Qs = self.get_Q_value_batch(self.network, next_state)
        target_Qs : torch.Tensor = self.network(next_state)
        best_action : list[int] = torch.argmax(target_Qs,dim=1).tolist()
        next_Q : torch.Tensor = self.network(next_state)[torch.arange(0, self.mem_batch_size), best_action]
        if terminate[best_action]:
            return reward.float()
        else:
            return (reward + self.gamma * next_Q).float()
    
    def train(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray, next_action:int, terminate:bool) -> tuple:
        """
        Model learning/optimization
        """        
        super().train(state, action, reward, next_state, next_action, terminate)
        state, action, next_reward, next_state, _ , terminate = self.memory.recall()

        q_target = next_reward + self.gamma * self.get_Q_value_batch(self.network, next_state)
        q_estimate = self.get_Q_value_batch(self.network, state, action)
        # q_est = self.current_q_w_estimate(state, action)
        # q_tgt = self.q_target(reward, next_state, terminate)
        loss = self.update_network(q_estimate, q_target)
        
        return (q_estimate.mean().item(), loss)


class DDQN(DeepAgent):
    def __init__(self,**kwargs:Unpack[AgentParams]) -> None:
        super().__init__(agent_type=AgentType.DDQN, **kwargs)
        self.target_net = self.net_type(kwargs['state_shape'], kwargs['num_actions'])
        # Copy inital weights from Q Network into the target network
        self.target_net.load_state_dict(self.network.state_dict())
        # Q_target parameters are frozen.
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.selection_network = self.network
        self.evaluation_network = self.target_net
        
    @torch.no_grad() # No Backwards computations needed
    def q_target(self, reward:torch.Tensor, next_state:torch.Tensor, terminate:bool) -> torch.Tensor:
        target_Qs : torch.Tensor = self.network(next_state)
        best_action : list[int] = torch.argmax(target_Qs,dim=1).tolist()
        next_Q : torch.Tensor = self.target_net(next_state)[torch.arange(0, self.mem_batch_size), best_action]
        if terminate[best_action]:
            return reward.float()
        else:
            return (reward + self.gamma * next_Q).float()
    
    def sync_w_to_target_net(self) -> None:
        self.target_net.load_state_dict(self.network.state_dict())

    def get_action_by_policy(self, state: np.ndarray, available_moves: list[int]) -> int:
        if random.random() < .5:
            self.selection_network = self.network
            self.evaluation_network = self.target_net
        else:
            self.selection_network = self.target_net
            self.evaluation_network = self.target_net
        return super().get_action_by_policy_(state, available_moves, self.selection_network)
        
    def train(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray, next_action:int, terminate:bool) -> tuple:
        """
        Model learning/optimization
        """
        if self.step % self.sync_interval == 0:
            self.sync_w_to_target_net()
        
        super().train(state, action, reward, next_state, next_action, terminate)

        state, action, reward, next_state, _, terminate = self.memory.recall()

        q_target = reward + self.gamma * self.get_Q_value_batch(self.evaluation_network, next_state)
        q_estimate = self.get_Q_value_batch(self.evaluation_network, state, action)
        loss = self.update_network(q_estimate, q_target)
        
        return (q_estimate.mean().item(), loss)

class DuelDQN(DeepAgent):
    def __init__(self,**kwargs:Unpack[AgentParams]) -> None:
        super().__init__(agent_type=AgentType.DUELDQN, **kwargs)
        
        self.value_net = self.net_type(kwargs['state_shape'], 1)
        self.advantage_net = self.network
        
    def current_q_w_estimate(self, state:np.ndarray, action:torch.Tensor) -> torch.Tensor:
        value : torch.Tensor = self.value_net(state)[np.arange(0, self.mem_batch_size), 0]
        advantages : torch.Tensor = self.network(state)[np.arange(0, self.mem_batch_size),:]
        mean_advantage = advantages.mean(dim=1)
        current_Q = value + (advantages[np.arange(0, self.mem_batch_size), action.tolist()] - mean_advantage)
        return current_Q
    
    def q_target(self, reward:torch.Tensor, next_state:torch.Tensor, terminate:bool) -> torch.Tensor:
        target_advantages : torch.Tensor = self.advantage_net(next_state)
        next_Q = torch.max(target_advantages,dim=1).values
        if terminate:
            return reward.float()
        else:
            return (reward + self.gamma * next_Q).float()

    def train(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray, next_action:int, terminate:bool) -> tuple:
        """
        Model learning/optimization
        """
        super().train(state, action, reward, next_state, next_action, terminate)
        state, action, reward, next_state, _ , terminate  = self.memory.recall()
        q_est = self.current_q_w_estimate(state, action)
        q_tgt = self.q_target(reward, next_state, terminate)
        loss = self.update_network(q_est, q_tgt)
        
        return (q_est.mean().item(), loss)
