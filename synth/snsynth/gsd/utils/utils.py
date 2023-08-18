import numpy as np

def get_sigma(rho: float,  sensitivity: float) -> float:
    if rho is None:
        return 0.0
    return np.sqrt(sensitivity**2 / rho)

def _divide_privacy_budget(rho: float, t: int) -> float:
    if rho is None: return None
    return rho / t
