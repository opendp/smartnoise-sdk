class SDGYMBaseSynthesizer:
    """
    Base for SmartNoise Synthesizers, based off of SDGymBaseSynthesizer
    (to allow for benchmarking)
    """

    def fit(self, data, categorical_columns=None, ordinal_columns=None):
        """
        Fit the synthesizer model on the data.

        Parameters
        ----------
        data : pd.DataFrame
            The data for fitting the synthesizer model.

        categorical_columns : list[str]
            List of column names for categorical columns

        ordinal_columns : list[str]
            List of column names for ordinal columns

        Returns
        -------
        pd.DataFrame
            Dataframe containing the generated data samples.
        """
        pass

    def sample(self, samples, categorical_columns=None, ordinal_columns=None):
        """
        Sample from the synthesizer model.

        Parameters
        ----------
        samples : int
            The number of samples to create

        categorical_columns : list[str]
            List of column names for categorical columns

        ordinal_columns : list[str]
            List of column names for ordinal columns

        Returns
        -------
        pd.Dataframe
            Dataframe containing the generated data samples.
        """
        pass

    def fit_sample(self, data, categorical_columns=None, ordinal_columns=None):
        """
        Fit the synthesizer model and then generate a synthetic dataset of the same
        size of the input data.

        Parameters
        ----------
        data : pd.DataFrame
            The data for fitting the synthesizer model.

        categorical_columns : list[str]
            List of column names for categorical columns

        ordinal_columns : list[str]
            List of column names for ordinal columns

        Returns
        -------
        pd.DataFrame
            Dataframe containing the generated data samples.
        """
        self.fit(data, categorical_columns, ordinal_columns)
        return self.sample(data.shape[0])
