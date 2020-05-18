class SDGYMBaseSynthesizer():
    """
    Base for Whitenoise Synthesizers, based off of SDGymBaseSynthesizer
    (to allow for benchmarking)
    """

    def fit(self, data):
        """
        Fits some data to synthetic data model.
        """
        pass

    def sample(self, samples):
        """
        Produces n (samples) using the fitted synthetic data model.
        """
        pass

    def fit_sample(self, data):
        """
        Common use case. Fits a synthetic data model to data, and returns
        # of samples equal to size of original dataset.
        Note data must be numpy array.
        """
        self.fit(data)
        return self.sample(data.shape[0])