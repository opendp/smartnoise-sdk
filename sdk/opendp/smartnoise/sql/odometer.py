import numpy as np
from opendp.smartnoise.sql.privacy import Privacy

class Odometer:
    def __init__(self, privacy: Privacy):
        Privacy
        self.k = 0
        self.privacy = privacy
        if not self.privacy.delta:
            self.privacy.delta = 0.0
        self.tol = self.privacy.delta / 2
    def spend(self, k=1):
        self.k += k
    def reset(self):
        self.k = 0
    @property
    def spent(self):
        """
        Implements k-folds homogeneous composition from Kairouz, et al
        https://arxiv.org/pdf/1311.0776.pdf
        """
        epsilon = self.privacy.epsilon
        delta = self.privacy.delta
        tol = self.tol

        if self.k == 0:
            return (0.0, 0.0)

        basic = self.k * epsilon
        optimal_left_side = ((np.exp(epsilon) - 1) * epsilon * self.k)/(np.exp(epsilon) + 1)
        optimal_a = optimal_left_side + epsilon * np.sqrt(2 * self.k * np.log(epsilon + (np.sqrt(self.k*epsilon*epsilon)/tol)))
        optimal_b = optimal_left_side + epsilon * np.sqrt(2 * self.k * (1/tol))
        delta = 1 - (1 - delta) ** self.k
        delta = delta * (1 - delta) + self.tol
        return tuple([min(basic, optimal_a, optimal_b), delta])