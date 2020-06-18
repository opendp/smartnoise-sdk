import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import normal
form torch.optim import SGD, Adam, Adagrad, RMSprop