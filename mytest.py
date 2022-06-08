import torch

x = [torch.randn((2, 4))]
y = [torch.randn((2, 4))]
print(x)
print(y)

x += [y]
print(x)

x = torch.cat(x, 0)
print(x)
