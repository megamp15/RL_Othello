# Install Pytorch Cuda: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# From https://pytorch.org/get-started/locally/ selected: Stable, Windows, Pip, Python, CUDA 12.1
# CUDA SEMANTICS: https://pytorch.org/docs/stable/notes/cuda.html
# Also need to install visual studio before the above. 

import torch

print(torch.cuda.is_available())


print(torch.cuda.device_count())

print(torch.cuda.current_device())


print(torch.cuda.device(0))

print(torch.cuda.get_device_name(0))
