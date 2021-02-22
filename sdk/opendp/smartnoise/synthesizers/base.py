class SDGYMBaseSynthesizer:
    """
    Base for SmartNoise Synthesizers, based off of SDGymBaseSynthesizer
    (to allow for benchmarking)
    """

    def fit(self, data, categorical_columns=None, ordinal_columns=None):
        """
        Fits the synthetic data approach on the provided data.
        """
        pass

    def sample(self, samples, categorical_columns=None, ordinal_columns=None):
        """
        Produces n (samples) using the fitted synthetic data model.
        """
        pass

    def fit_sample(self, data, categorical_columns=None, ordinal_columns=None):
        """
        Common use case. Fits a synthetic data model to data, and returns
        # of samples equal to size of original dataset.
        Note data must be numpy array.
        """
        self.fit(data, categorical_columns, ordinal_columns)
        return self.sample(data.shape[0])
