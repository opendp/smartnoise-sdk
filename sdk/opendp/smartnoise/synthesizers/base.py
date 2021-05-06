class SDGYMBaseSynthesizer:

    def fit(self, data, categorical_columns=None, ordinal_columns=None):
        """
        Fit the synthesizer model on the data.

        :param data: The data for fitting the synthesizer model.
        :type data: pd.DataFrame
        :param categorical_columns: List of column names for categorical columns, defaults to None
        :type categorical_columns: list[str], optional
        :param ordinal_columns: List of column names for ordinal columns, defaults to None
        :type ordinal_columns: list[str], optional
        :return: Dataframe containing the generated data samples.
        :rtype: pd.DataFrame
        """
        pass

    def sample(self, samples, categorical_columns=None, ordinal_columns=None):
        """
        Sample from the synthesizer model.

        :param samples: The number of samples to create
        :type samples: int
        :param categorical_columns: List of column names for categorical columns, defaults to None
        :type categorical_columns: list[str], optional
        :param ordinal_columns: List of column names for ordinal columns, defaults to None
        :type ordinal_columns: list[str], optional
        :return: Dataframe containing the generated data samples.
        :rtype: pd.DataFrame
        """
        pass

    def fit_sample(self, data, categorical_columns=None, ordinal_columns=None):
        """
        Fit the synthesizer model and then generate a synthetic dataset of the same
        size of the input data.

        :param data: The data for fitting the synthesizer model.
        :type data: pd.DataFrame
        :param categorical_columns: List of column names for categorical columns, defaults to None
        :type categorical_columns: list[str], optional
        :param ordinal_columns: List of column names for ordinal columns, defaults to None
        :type ordinal_columns: list[str], optional
        :return: Dataframe containing the generated data samples.
        :rtype: pd.DataFrame
        """
        self.fit(data, categorical_columns, ordinal_columns)
        return self.sample(data.shape[0])
