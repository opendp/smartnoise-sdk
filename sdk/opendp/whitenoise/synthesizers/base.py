class SDGYMBaseSynthesizer():
    """
    Base for Whitenoise Synthesizers, based off of SDGymBaseSynthesizer
    (to allow for benchmarking)
    """

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        """
        Fits some data to synthetic data model.
        """
        pass

    def sample(self, samples):
        """
        Produces n (samples) using the fitted synthetic data model.
        """
        pass

    def fit_sample(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        """
        Common use case. Fits a synthetic data model to data, and returns
        # of samples equal to size of original dataset.
        Note data must be numpy array.
        """
        self.fit(data, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns)
        return self.sample(data.shape[0])