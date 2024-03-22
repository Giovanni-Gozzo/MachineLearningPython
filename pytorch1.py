import torch
import numpy as np

list= np.random.rand(3,4)
print(list)

tensor_2d= torch.randn(3,4)
print(tensor_2d)

tensor_3d= torch.zero_(2,3,4)
print(tensor_3d)

tensor_np= torch.tensor(list)
print(tensor_np)
