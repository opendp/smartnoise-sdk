import torch
import torch.nn as nn
import math
import numpy as np


def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def pate(data, teachers, lap_scale, device="cpu"):
    """PATE implementation for GANs.
    """
    num_teachers = len(teachers)
    labels = torch.Tensor(num_teachers, data.shape[0]).type(torch.int64).to(device)
    for i in range(num_teachers):
        output = teachers[i](data)
        pred = (output > 0.5).type(torch.Tensor).squeeze().to(device)
        # print(pred.shape)
        # print(labels[i].shape)
        labels[i] = pred

    votes = torch.sum(labels, dim=0).unsqueeze(1).type(torch.DoubleTensor).to(device)
    noise = torch.from_numpy(np.random.laplace(loc=0, scale=1 / lap_scale, size=votes.size())).to(
        device
    )
    noisy_votes = votes + noise
    noisy_labels = (noisy_votes > num_teachers / 2).type(torch.DoubleTensor).to(device)

    return noisy_labels, votes


def moments_acc(num_teachers, votes, lap_scale, l_list, device="cpu"):
    q = (2 + lap_scale * torch.abs(2 * votes - num_teachers)) / (
        4 * torch.exp(lap_scale * torch.abs(2 * votes - num_teachers))
    ).to(device)

    alpha = []
    for l_val in l_list:
        a = 2 * lap_scale ** 2 * l_val * (l_val + 1)
        t_one = (1 - q) * torch.pow((1 - q) / (1 - math.exp(2 * lap_scale) * q), l_val)
        t_two = q * torch.exp(2 * lap_scale * l_val)
        t = t_one + t_two
        alpha.append(torch.clamp(t, max=a).sum())

    return torch.DoubleTensor(alpha).to(device)
