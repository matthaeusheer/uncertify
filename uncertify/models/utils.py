import math

"""Code taken from https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/7"""


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """Calculates the output height and width of a feature map for a Conv2D operation."""
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), num2tuple(kernel_size), num2tuple(stride), \
                                              num2tuple(pad), num2tuple(dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])
    out_height = math.floor((h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    out_width = math.floor((h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return out_height, out_width


def convtranspose2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0):
    """Calculates the output height and width of a feature map for a ConvTranspose2D operation."""
    h_w, kernel_size, stride, pad, dilation, out_pad = num2tuple(h_w), num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation), num2tuple(out_pad)
    pad = num2tuple(pad[0]), num2tuple(pad[1])
    out_height = (h_w[0] - 1) * stride[0] - sum(pad[0]) + dilation[0] * (kernel_size[0] - 1) + out_pad[0] + 1
    out_width = (h_w[1] - 1) * stride[1] - sum(pad[1]) + dilation[1] * (kernel_size[1] - 1) + out_pad[1] + 1
    return out_height, out_width
