import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import numpy as np

class ReplayMemory():
    """
    Different replay buffers in PyTorch
    https://pytorch.org/rl/tutorials/rb_tutorial.html
    """
    def __init__(self, capacity, batch_size, device):
        self.device = device
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(capacity, device=device))
        self.batch_size = batch_size

    def cache(self, state:np.ndarray, action:int, reward:int, next_state:np.ndarray, terminate:int, next_action:int=None):
        """
        Add (s, a, r, s') to memory
        """
        data = TensorDict( {
            "state": torch.tensor(state, dtype=torch.float, device=self.device),
            "action": torch.tensor([action], dtype=torch.long, device=self.device),
            "reward": torch.tensor([reward], dtype=torch.float, device=self.device),
            "next_state": torch.tensor(next_state, dtype=torch.float, device=self.device),
            "terminate": torch.tensor([terminate], dtype=torch.float, device=self.device)
        }, batch_size=[])
        if next_action is not None: 
            data["next_action"] = torch.tensor([next_action], device=self.device)
        self.memory.add(data)

    def recall(self):
        """
        Retrieve experience from memory
        """
        samples = self.memory.sample(self.batch_size).to(self.device)
        state, action, reward, next_state, terminate = (samples.get(key) for key in ("state", "action", "reward", "next_state", "terminate"))
        state=state.detach().clone()
        action=action.detach().clone()
        reward=reward.detach().clone() 
        next_state=next_state.detach().clone()
        terminate=terminate.detach().clone()
        if "next_action" in samples.keys():
            next_action = samples["next_action"].detach().clone()
            return state, action.squeeze(), reward.squeeze(), next_state, next_action.squeeze(), terminate.squeeze()

        return state, action.squeeze(), reward.squeeze(), next_state, terminate.squeeze()

    def __len__(self):
        return len(self.memory)