import numpy as np


def evaluate(sens, epsilon):
    scale = sens / epsilon
    noise = np.random.laplace(scale=scale, size=1)
    return noise[0]
