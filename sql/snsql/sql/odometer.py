import numpy as np
from snsql.sql.privacy import Privacy

class Odometer:
    """
    Implements k-folds homogeneous composition from Kairouz, et al
    Theorem 3.4
    https://arxiv.org/pdf/1311.0776.pdf
    """
    def __init__(self, privacy: Privacy):
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

class OdometerHeterogeneous:
    """
    Implements k-folds heterogeneous composition from Kairouz, et al
    Theorem 3.5
    https://arxiv.org/pdf/1311.0776.pdf
    """
    def __init__(self, privacy: Privacy):
        self.steps = []
        self.privacy = privacy
        self.tol = None
        if privacy:
            if not self.privacy.delta:
                self.privacy.delta = 0.0
            self.tol = self.privacy.delta / 2
        if self.tol == 0.0:
            self.tol = 10E-16
    def spend(self, privacy: Privacy = None):
        if privacy:
            if not self.tol:
                self.tol = privacy.delta / 2
            if self.tol > privacy.delta and privacy.delta != 0.0:
                self.tol = privacy.delta
            self.steps.append((privacy.epsilon, privacy.delta))
        elif self.privacy:
            self.steps.append((self.privacy.epsilon, self.privacy.delta))
        else:
            raise ValueError("No privacy information passed in")
    def reset(self):
        self.steps = []
    @property
    def k(self):
        return len(self.steps)
    @property
    def spent(self):
        if self.steps == []:
            return (0.0, 0.0)

        # delta
        delta = 1 - (1 - self.tol) * np.prod([(1 - delta) for _, delta in self.steps])

        # epsilon
        basic = np.sum([eps for eps, _ in self.steps])
        optimal_left_side = np.sum([((np.exp(eps) - 1) * eps) / ((np.exp(eps) + 1)) for eps, _ in self.steps])
        sq = np.sum([eps * eps for eps, _ in self.steps])
        sqsq = np.sum([2 * eps * eps for eps, _ in self.steps])
        optimal_a = optimal_left_side + np.sqrt(sqsq * np.log(np.exp(1) + (np.sqrt(sq)/self.tol)))
        optimal_b = optimal_left_side + np.sqrt(sqsq * np.log(1/self.tol))
        
        return tuple([min(basic, optimal_a, optimal_b), delta])
