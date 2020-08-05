import torch
from torch import nn
import torch.nn.functional as F


class Gradient(nn.Module):
    """Gradient calculator for single-channel image batches."""
    def __init__(self):
        super().__init__()
        kernel_x = [[-1., 0., 1.],
                    [-2., 0., 2.],
                    [-1., 0., 1.]]
        kernel_x = torch.Tensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = [[-1., 0., 1.],
                    [-2., 0., 2.],
                    [-1., 0., 1.]]
        kernel_y = torch.Tensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x.float(), self.weight_x.float(), padding=1)  # floats needed for apex 16bit precision...?
        grad_y = F.conv2d(x.float(), self.weight_y.float(), padding=1)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient
