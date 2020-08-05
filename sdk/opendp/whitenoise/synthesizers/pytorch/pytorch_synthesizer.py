import numpy as np
import pandas as pd

from opendp.whitenoise.synthesizers.preprocessors.preprocessing import GeneralTransformer
from opendp.whitenoise.synthesizers.base import SDGYMBaseSynthesizer

class PytorchSynthesizer(SDGYMBaseSynthesizer):
    def __init__(self, gan, preprocessor):
        self.gan = gan
        self.preprocessor = preprocessor

        self.categorical_columns = None
        self.ordinal_columns = None
    
    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.categorical_columns = categorical_columns
        self.ordinal_columns = ordinal_columns
        
        if self.preprocessor is not None:
            self.preprocessor.fit(data, categorical_columns, ordinal_columns)
            preprocessed_data = self.preprocessor.transform(data)
            self.gan.train(preprocessed_data)
        else:
            self.gain.train(data)
    
    def sample(self, n):
        synth_data = self.gan.generate(n)
        
        if self.preprocessor is not None:
            if isinstance(self.preprocessor, GeneralTransformer):
                synth_data = self.preprocessor.inverse_transform(synth_data, None)
            else:
                synth_data = self.preprocessor.inverse_transform(synth_data)

        return synth_data