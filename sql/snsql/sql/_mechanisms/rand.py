import random
import math

sys_rand = random.SystemRandom()


def normal(loc, scale, size):
    return [sys_rand.gauss(loc, scale) for i in range(size)]


def laplace(loc, scale, size):
    U = [sys_rand.uniform(-0.5, 0.5) for i in range(size)]
    return [loc - scale * math.copysign(math.log(1 - (2 * abs(u))), u) for u in U]
