class Metrics:
    """
	Defines the fields available in the metrics payload object
	"""

    def __init__(self):
        self.dp_res = False
        self.wasserstein_distance = 0.0
        self.jensen_shannon_divergence = 0.0
        self.kl_divergence = 0.0
        self.mse = 0.0
        self.msd = 0.0
        self.std = 0.0
        self.acc_res = False
        self.within_bounds = 0
        self.outside_bounds = 0
        self.utility_res = False
        self.bias_res = False
        self.error = None
