from typing_extensions import Unpack
import torch
import torch.nn as nn
import numpy as np

from agent import DeepAgent, AgentType, AgentParams

class SARSA(DeepAgent):
    """
    A deep sarsa learning network agent
    """
    name = 'Sarsa'

    def __init__(self,**kwargs:Unpack[AgentParams]):
        super().__init__(agent_type=AgentType.SARSA, **kwargs)

    @torch.no_grad() # No Backwards computations needed
    def q_target(self, reward:torch.Tensor, next_state:torch.Tensor, next_action:torch.Tensor, terminate:torch.Tensor) -> float:
        target_Qs = self.network(next_state)
        next_Q = self.network(next_state)[
            torch.arange(0, self.mem_batch_size), next_action
        ]
        not_done = 1 - terminate # Invert for mult below
        return (reward + self.gamma * next_Q*not_done).float()
    
    def train(self, state, action, reward, next_state, next_action, terminate) -> tuple:
        """
        Model learning/optimization
        """        
        super().train(state, action, reward, next_state, next_action, terminate)
        state, action, reward, next_state, next_action, terminate = self.memory.recall()
        q_est = self.current_q_w_estimate(state, action)
        q_tgt = self.q_target(reward, next_state, next_action, terminate)
        loss = self.update_network(q_est, q_tgt)
        
        return (q_est.mean().item(), loss)

class SARSA_DDQN(DeepAgent):
    name = 'Sarsa_DDQN'

    def __init__(self,**kwargs:Unpack[AgentParams]):
        super().__init__(agent_type=AgentType.SARSA,**kwargs)
        self.target_net = self.net_type(kwargs['state_shape'], kwargs['num_actions'])
        # Copy inital weights from Q Network into the target network
        self.target_net.load_state_dict(self.network.state_dict())
        
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
        
    def train(self, state, action, reward, next_state, next_action, terminate) -> tuple:
        """
        Model learning/optimization
        """
        if self.step % self.sync_interval == 0:
            self.sync_w_to_target_net()
        
        super().train(state, action, reward, next_state, next_action, terminate)
        state, action, reward, next_state, next_action, terminate = self.memory.recall()
        q_est = self.current_q_w_estimate(state, action)
        q_tgt = self.q_target(reward, next_state, next_action, terminate)
        loss = self.update_network(q_est, q_tgt)
        
        return (q_est.mean().item(), loss)

class SARSA_DuelDQN(DeepAgent):
    name = 'Sarsa_DuelDQN'

    def __init__(self, **kwargs:Unpack[AgentParams]):
        super().__init__(AgentType.SARSA, **kwargs)
        
        self.value_net = self.net_type(kwargs['state_shape'], 1)
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
    
    def train(self, state, action, reward, next_state, next_action, terminate) -> tuple:
        """
        Model learning/optimization
        """        
        super().train(state, action, reward, next_state, next_action, terminate)
        state, action, reward, next_state, next_action, terminate = self.memory.recall()
        q_est = self.current_q_w_estimate(state, action)
        q_tgt = self.q_target(reward, next_state, next_action, terminate)
        loss = self.update_network(q_est, q_tgt)
        
        return (q_est.mean().item(), loss)
