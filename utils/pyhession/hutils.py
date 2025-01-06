

import torch
import math
from torch.autograd import Variable
import numpy as np





def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    """
    设置 retain_graph=True，保留了 gradsH 的计算图，使得：
    当前操作可以顺利完成。
    如果之后还有其他类似操作（如再次计算 Hessian 向量积），也不会出错。
    """
    return hv


def smooth_max(x, tau=1e-2):
    weights = torch.softmax(x / tau, dim=0)
    return torch.sum(weights * x)

def smooth_min(x, tau=1e-2):
    weights = torch.softmax(-x / tau, dim=0)
    return torch.sum(weights * x)
