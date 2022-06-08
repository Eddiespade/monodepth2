import torch

x = torch.randn((2, 4, 7, 7))
print(x.shape)
x = torch.nn.AvgPool2d(3, 1)(x)
print(x.shape)
