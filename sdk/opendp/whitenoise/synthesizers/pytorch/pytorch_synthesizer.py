import numpy as np
import pandas as pd

from opendp.whitenoise.synthesizers.preprocessors.preprocessing import GeneralTransformer
from opendp.whitenoise.synthesizers.base import SDGYMBaseSynthesizer

class PytorchDPSynthesizer(SDGYMBaseSynthesizer):
    def __init__(self, preprocessor, gan):
        self.preprocessor = preprocessor
        self.gan = gan

        self.categorical_columns = None
        self.ordinal_columns = None
    
    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.categorical_columns = categorical_columns
        self.ordinal_columns = ordinal_columns
        
        self.preprocessor.fit(data, categorical_columns, ordinal_columns)
        preprocessed_data = self.preprocessor.transform(data)

        self.gan.train(preprocessed_data)
    
    def sample(self, n):
        synth_data = self.gan.generate(n)
        
        if isinstance(self.preprocessor, GeneralTransformer):
            synth_data = self.preprocessor.inverse_transform(synth_data, None)
        else:
            synth_data = self.preprocessor.inverse_transform(synth_data)

        return synth_data