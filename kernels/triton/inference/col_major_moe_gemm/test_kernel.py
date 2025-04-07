import torch
torch.cuda.init()
a = torch.randn(1024, 1024, device='cuda')  # Create a 1024x1024 tensor on the GPU
b = torch.randn(1024, 1024, device='cuda')  # Another 1024x1024 tensor on the GPU
c = a @ b  # Perform matrix multiplication, launching a GPU kernel
