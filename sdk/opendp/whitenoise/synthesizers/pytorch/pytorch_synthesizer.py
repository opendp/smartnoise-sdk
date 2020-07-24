import numpy as np
import pandas as pd

from opendp.whitenoise.synthesizers.preprocessors.preprocessing import GeneralTransformer
from opendp.whitenoise.synthesizers.base import SDGYMBaseSynthesizer

class PytorchDPSynthesizer(SDGYMBaseSynthesizer):
    def __init__(self, preprocessor, gan, epsilon):
        self.preprocessor = preprocessor
        self.gan = gan

        self.categorical_columns = None
        self.ordinal_columns = None

        self.epsilon = epsilon

        self.pd_cols = None
        self.pd_index = None
        self.pandas = False

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.categorical_columns = categorical_columns
        self.ordinal_columns = ordinal_columns

        if isinstance(data, pd.DataFrame):
            self.pandas = True
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='ignore')
            self.pd_cols = data.columns
            self.pd_index = data.index

        self.preprocessor.fit(data, categorical_columns, ordinal_columns)
        preprocessed_data = self.preprocessor.transform(data)

        self.gan.train(preprocessed_data, epsilon=self.epsilon)

    def sample(self, n):
        synth_data = self.gan.generate(n)

        if isinstance(self.preprocessor, GeneralTransformer):
            synth_data = self.preprocessor.inverse_transform(synth_data, None)
        else:
            synth_data = self.preprocessor.inverse_transform(synth_data)

        if self.pandas:
            df = pd.DataFrame(synth_data, 
                index = self.pd_index,
                columns = self.pd_cols)
            return df

        return synth_data 