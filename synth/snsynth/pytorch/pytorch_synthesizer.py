from functools import wraps

import numpy as np
import pandas as pd
import warnings

from snsynth.base import SDGYMBaseSynthesizer


class PytorchDPSynthesizer(SDGYMBaseSynthesizer):
    def __init__(self, epsilon, gan, preprocessor=None):
        """
        Wrapper class to unify pytorch GAN architectures with the SDGYM API.

        :param epsilon: Total epsilon used for the DP Synthesizer
        :type epsilon: float
        :param gan: A pytorch defined GAN
        :type gan: torch.nn.Module
        :param preprocessor: A preprocessor to .transform the input data and
            .inverse_transform the output of the GAN., defaults to None
        :type preprocessor: GeneralTransformer, optional
        """
        self.epsilon = epsilon
        self.gan = gan
        self.preprocessor = preprocessor

        self._data_columns = None

    def _get_training_data(self, data, categorical_columns, ordinal_columns):
        if not self.preprocessor:
            return data
        else:
            self.preprocessor.fit(data, categorical_columns, ordinal_columns)
            return self.preprocessor.transform(data)

    @wraps(SDGYMBaseSynthesizer.fit)
    def fit(
        self,
        data,
        categorical_columns=tuple(),
        ordinal_columns=tuple(),
        transformer=None,
        continuous_columns=None,
        preprocessor_eps=0.0,
        nullable=False,
    ):
        def column_names(n_items, prefix="col"):
            names = []
            for i in range(n_items):
                names.append(prefix + "_" + str(i))
            return names

        if isinstance(data, pd.DataFrame):
            self._data_columns = data.columns
        elif isinstance(data, np.ndarray):
            placeholder_columns = column_names(data.shape[1])
            data = pd.DataFrame(data, columns=placeholder_columns).infer_objects()
            self._data_columns = placeholder_columns
            warnings.warn(
                "Data is numpy array, converting to pandas dataframe with default "
                + "column names. Inferring data types. Note: for best performance, "
                + "pandas dataframe should be constructed by user and "
                + "data_types should be specified beforehand. Dtypes: "
                + str(data.dtypes),
                Warning,
            )

        self.dtypes = data.dtypes

        training_data = self._get_training_data(
            data, categorical_columns, ordinal_columns
        )

        self.gan.train(
            training_data,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
            update_epsilon=self.epsilon,
            transformer=transformer,
            continuous_columns=continuous_columns,
            preprocessor_eps=preprocessor_eps,
            nullable=nullable,
        )

    @wraps(SDGYMBaseSynthesizer.sample)
    def sample(self, n):
        synth_data = self.gan.generate(n)

        if self.preprocessor is not None:
            synth_data = self.preprocessor.inverse_transform(synth_data)

        if isinstance(synth_data, np.ndarray):
            synth_data = pd.DataFrame(synth_data, columns=self._data_columns)
        elif isinstance(synth_data, pd.DataFrame):
            # TODO: Add validity check
            synth_data.columns = self._data_columns
        else:
            raise ValueError("Generated data is neither numpy array nor dataframe!")

        return synth_data
