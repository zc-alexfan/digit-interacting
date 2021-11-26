import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .ld_utils import unsort as unsort_list


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N, D1, D2, ..].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, D1, D2, .., Dk, #classes].
    """
    y = torch.eye(num_classes).float()
    return y[labels]

def unsort(ten, sort_idx):
    """
    Unsort a tensor of shape (N, *) using the sort_idx list(N).
    Return a tensor of the pre-sorting order in shape (N, *)
    """
    assert isinstance(ten, torch.Tensor)
    assert isinstance(sort_idx, list)
    assert ten.shape[0] == len(sort_idx)

    out_list = list(torch.chunk(ten, ten.size(0), dim=0))
    out_list = unsort_list(out_list, sort_idx)
    out_list = torch.cat(out_list, dim=0)
    return out_list


def softargmax_kd(inputs_hm, temperature):
    """
    Perform softargmax on a k-dimensional heatmap
    Inputs: (N, D1, D2, .., Dk)
    Outputs: (N, K)
    """
    assert isinstance(inputs_hm, torch.Tensor)
    assert isinstance(temperature, float)
    assert len(inputs_hm.shape) >= 2
    dev = inputs_hm.device
    batch_size = inputs_hm.shape[0]
    num_dim = len(inputs_hm.shape) - 1
    dims = inputs_hm.shape[1:]
    lins = [torch.arange(dim).view(-1, 1).to(dev) + 0.5 for dim in dims]
    points_kd = lins[0]
    for lin in lins[1:]:
        points_kd = all_comb(points_kd, lin)

    points_kd = points_kd.unsqueeze(0)
    hms = inputs_hm.view(batch_size, -1, 1)

    norm_hms = F.softmax(hms*temperature, dim=1)
    weighted_points = (points_kd*norm_hms).sum(dim=1)
    assert weighted_points.shape[0] == batch_size
    assert weighted_points.shape[1] == num_dim
    return weighted_points


def gumbel_sample_kd(hm, num_samples, tau, logit_scale, hard):
    """
    hm: (N, D1, D2, D3, .., Dk), e.g., (N, x, y, z)
    num_samples: int, num samples to draw from each heatmap (N*samples) samples at the end
    tau: float, tau term for gumbel
    logit_scale: scale back logits before log(logits)
    hard: gives one hot softmax prob or not in gumbel.

    Outputs:
        samples: (N, samples, k)
    """
    batch_size = hm.shape[0]
    hm_size = hm.shape[1]
    assert np.all(np.array(hm.shape[1:]) == hm_size)
    num_dim = len(hm.shape[1:])
    dev = hm.device

    assert num_dim == 2 or num_dim == 3, "This function only supports 2d/3d heatmaps; verify for higher dim"
    hm = hm.reshape(batch_size, 1, -1)
    prob_dim = hm.shape[2]
    mesh_grids = torch.meshgrid([torch.arange(hm_size) for _ in range(num_dim)])
    mesh_grids = torch.stack(mesh_grids, dim=-1).to(dev).float()
    mesh_grids = mesh_grids.view(1, 1, prob_dim, num_dim)

    hm = hm.expand(batch_size, num_samples, prob_dim).reshape(batch_size*num_samples, prob_dim)
    samples = F.gumbel_softmax(logit_scale*hm, tau=tau, hard=hard, dim=1)
    samples = samples.view(batch_size, num_samples, prob_dim, 1)
    samples = samples*mesh_grids
    samples = samples.sum(dim=2)
    return samples


def gumbel_sample_kd_iter(hm, num_samples, tau, logit_scale, hard, chunk_size):
    """
    Helper for gumbel_sample_kd to avoid memory requirements by splitting the batch into chunks
    hm: (N, D1, D2, D3, .., Dk), e.g., (N, x, y, z)
    num_samples: int, num samples to draw from each heatmap (N*samples) samples at the end
    tau: float, tau term for gumbel
    logit_scale: scale back logits before log(logits)
    hard: gives one hot softmax prob or not in gumbel.

    Outputs:
        samples: (N, samples, k)
    """
    hm_splits = list(hm.split(chunk_size))
    sample_list = []
    for curr_hm in hm_splits:
        sample_list.append(gumbel_sample_kd(
            curr_hm, num_samples, tau, logit_scale, hard))
    samples = torch.cat(sample_list, dim=0)
    return samples


def fetch_comb_index(num, dev, comb_type):
    if comb_type == 'lower':
        comb_idx = torch.LongTensor([
            (i, j) for i in range(num) for j in range(num)
            if i < j]).to(dev)
    elif comb_type == 'lower_diag':
        comb_idx = torch.LongTensor([
            (i, j) for i in range(num) for j in range(num)
            if i <= j]).to(dev)
    elif comb_type == 'diag':
        comb_idx = torch.LongTensor([
            (i, j) for i in range(num) for j in range(num)
            if i == j]).to(dev)
    elif comb_type == 'off_diag':
        comb_idx = torch.LongTensor([
            (i, j) for i in range(num) for j in range(num)
            if i != j]).to(dev)
    elif comb_type == 'full':
        comb_idx = torch.LongTensor([
            (i, j) for i in range(num) for j in range(num)
            ]).to(dev)
    else:
        assert False
    return comb_idx

def all_comb(X, Y):
    """
    Returns all possible combinations of elements in X and Y.
    X: (n_x, d_x)
    Y: (n_y, d_y)
    Output: Z: (n_x*x_y, d_x+d_y)
    Example:
    X = tensor([[8, 8, 8],
                [7, 5, 9]])
    Y = tensor([[3, 8, 7, 7],
                [3, 7, 9, 9],
                [6, 4, 3, 7]])
    Z = tensor([[8, 8, 8, 3, 8, 7, 7],
                [8, 8, 8, 3, 7, 9, 9],
                [8, 8, 8, 6, 4, 3, 7],
                [7, 5, 9, 3, 8, 7, 7],
                [7, 5, 9, 3, 7, 9, 9],
                [7, 5, 9, 6, 4, 3, 7]])
    """
    assert len(X.size()) == 2
    assert len(Y.size()) == 2
    X1 = X.unsqueeze(1)
    Y1 = Y.unsqueeze(0)
    X2 = X1.repeat(1, Y.shape[0], 1)
    Y2 = Y1.repeat(X.shape[0], 1, 1)
    Z = torch.cat([X2, Y2], -1)
    Z = Z.view(-1, Z.shape[-1])
    return Z

def toggle_parameters(model, requires_grad):
    """
    Set all weights to requires_grad or not.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def tensor2np(ten):
    """This function move tensor to cpu and convert to numpy"""
    return ten.cpu().detach().numpy()


def count_model_parameters(model):
    """
    Return the amount of parameters that requries gradients.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def list2dev(L, dev):
    assert isinstance(L, list)
    return [ten.to(dev) for ten in L]

def dict2dev(d, dev, selected_keys=None):
    """
    If selected_keys is not None, then only move to the device for those keys
    """
    types = (torch.Tensor, torch.nn.utils.rnn.PackedSequence)
    if selected_keys is None:
        selected_keys = list(d.keys())
    for k, v in d.items():
        if k in selected_keys:
            if isinstance(v, types):
                d[k] = d[k].to(dev)
            elif isinstance(v, list) and len(v) > 0\
                    and isinstance(v[0], types):
                d[k] = list2dev(v, dev)

    return d

def reset_all_seeds():
    """Reset all seeds for reproduciability."""
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)


def get_activation(name):
    """This function return an activation constructor by name."""
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'selu':
        return nn.SELU()
    elif name == 'relu6':
        return nn.ReLU6()
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'softshrink':
        return nn.Softshrink()
    else:
        print("Undefined activation: %s" % (name))
        assert False


def stack_ll_tensors(tensor_list_list):
    """
    Recursively stack a list of lists of lists .. whose elements are tensors with the same shape
    """
    if isinstance(tensor_list_list, torch.Tensor):
        return tensor_list_list
    assert isinstance(tensor_list_list, list)
    if isinstance(tensor_list_list[0], torch.Tensor):
        return torch.stack(tensor_list_list)

    stacked_tensor = []
    for tensor_list in tensor_list_list:
        stacked_tensor.append(stack_ll_tensors(tensor_list))
    stacked_tensor = torch.stack(stacked_tensor)
    return stacked_tensor


def get_optim(name):
    """This function return an optimizer constructor by name."""
    if name == 'adam':
        return optim.Adam
    elif name == 'rmsprop':
        return optim.RMSprop
    elif name == 'sgd':
        return optim.SGD
    else:
        print("Undefined optim: %s" % (name))
        assert False


def decay_lr(optimizer, gamma):
    """
    Decay the learning rate by gamma
    """
    assert isinstance(gamma, float)
    assert 0 <= gamma and gamma <= 1.0
    for param_group in optimizer.param_groups:
        param_group['lr'] *= gamma
