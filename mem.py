import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class ReplayMemory():
    """
    Different replay buffers in PyTorch
    https://pytorch.org/rl/tutorials/rb_tutorial.html
    """
    def __init__(self, capacity, batch_size, device):
        self.device = device
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(capacity, device=torch.device(device)))
        self.batch_size = batch_size

    def cache(self, state:torch.Tensor, action:int, reward:int, next_state:torch.Tensor, terminate:int, next_action:int=None):
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
        if next_action is not None: 
            data["next_action"] = torch.Tensor([next_action])
        self.memory.add(data)

    def recall(self):
        """
        Retrieve experience from memory
        """
        samples = self.memory.sample(self.batch_size).to(self.device)
        state, action, reward, next_state, terminate = (samples.get(key) for key in ("state", "action", "reward", "next_state", "terminate"))
        state=state.clone().detach().float()  
        action=action.clone().detach().long() 
        reward=reward.clone().detach().float() 
        next_state=next_state.clone().detach().float() 
        terminate=terminate.clone().detach().float() 

        if "next_action" in samples.keys():
            next_action = samples["next_action"].clone().detach().long() 
            return state, action.squeeze(), reward.squeeze(), next_state, next_action.squeeze(), terminate.squeeze()

        return state, action.squeeze(), reward.squeeze(), next_state, terminate.squeeze()

    def __len__(self):
        return len(self.memory)