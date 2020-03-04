import torch


def quadratic_loss(input, target):
    return torch.sum((input - target) ** 2) * 0.5
