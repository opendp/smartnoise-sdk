import numpy as np
import pandas as pd

from opendp.whitenoise.synthesizers.preprocessors.preprocessing import GeneralTransformer
from opendp.whitenoise.synthesizers.base import SDGYMBaseSynthesizer

class PytorchDPSynthesizer(SDGYMBaseSynthesizer):
    def __init__(self, gan, preprocessor=None):
        self.preprocessor = preprocessor
        self.gan = gan

        self.categorical_columns = None
        self.ordinal_columns = None
        self.dtypes = None
    
    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.categorical_columns = categorical_columns
        self.ordinal_columns = ordinal_columns
        self.dtypes = data.dtypes
        self.columns = data.columns


        if self.preprocessor:
            self.preprocessor.fit(data, categorical_columns, ordinal_columns)
            data = self.preprocessor.transform(data)

        self.gan.train(data, categorical_columns, ordinal_columns)
    
    def sample(self, n):
        synth_data = self.gan.generate(n)

        if self.preprocessor:
            if isinstance(self.preprocessor, GeneralTransformer):
                synth_data = self.preprocessor.inverse_transform(synth_data, None)
            else:
                synth_data = self.preprocessor.inverse_transform(synth_data)

        if isinstance(synth_data, np.ndarray):
            synth_data = pd.DataFrame(synth_data,columns=self.columns)

        elif isinstance(synth_data, pd.DataFrame):
            if sum(synth_data.dtypes!=self.dtypes) > 0 :
                convert_dict = self.dtypes.to_dict()
                synth_data = synth_data.astype(convert_dict)
        else:
            raise ValueError("generated data is neither numpy array nor dataframe! ")
            

        

        return synth_data


