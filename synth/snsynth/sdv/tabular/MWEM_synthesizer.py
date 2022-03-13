import numpy as np
from snsynth import MWEMSynthesizer

from snsynth.sdv.tabular.base import BaseTabularModel

# NOTE: Adding this as optional import, may need adjusting
try: 
    from sdv.metadata import Table
except ImportError: 
    Table = None


class SmartnoiseMWEMModel(BaseTabularModel):

    """Base class for all the CTGAN models.
    The ``CTGANModel`` class provides a wrapper for all the CTGAN models.
    """

    _MODEL_CLASS = None
    _model_kwargs = None

    _DTYPE_TRANSFORMERS = {
        'O': None
    }

    def _build_model(self):
        return self._MODEL_CLASS(**self._model_kwargs)
    
    def _fit(self, data):
        """Fit the model to the table.
        Args:
            data (pandas.DataFrame):
                Data to be learned.
        """
        self._model = self._build_model()

        categoricals = []
        fields_before_transform = self._metadata.get_fields()
        for field in data.columns:
            if field in fields_before_transform:
                meta = fields_before_transform[field]
                if meta['type'] == 'categorical':
                    categoricals.append(field)

            else:
                field_data = data[field].dropna()
                if set(field_data.unique()) == {0.0, 1.0}:
                    # booleans encoded as float values must be modeled as bool
                    field_data = field_data.astype(bool)

                dtype = field_data.infer_objects().dtype
                try:
                    kind = np.dtype(dtype).kind
                except TypeError:
                    # probably category
                    kind = 'O'
                if kind in ['O', 'b']:
                    categoricals.append(field)

        self._model.fit(
            data,
            categorical_columns=categoricals
        )

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.
        Args:
            num_rows (int):
                Amount of rows to sample.
        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        return self._model.sample(num_rows)

class MWEM(SmartnoiseMWEMModel):
    _MODEL_CLASS = MWEMSynthesizer

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 epsilon=3.0,
                 q_count=400,
                 iterations=30,
                 mult_weights_iterations=20,
                 splits=[],
                 split_factor=None,
                 max_bin_count=500,
                 custom_bin_count={},
                 max_retries_exp_mechanism=1000,
                 rounding='auto', 
                 min_value='auto', 
                 max_value='auto'):
        super().__init__(
            field_names=field_names,
            primary_key=primary_key,
            field_types=field_types,
            field_transformers=field_transformers,
            anonymize_fields=anonymize_fields,
            constraints=constraints,
            table_metadata=table_metadata,
            rounding=rounding,
            max_value=max_value,
            min_value=min_value
        )

        self._model_kwargs = {
            'epsilon': epsilon,
            'q_count': q_count,
            'iterations': iterations,
            'mult_weights_iterations': mult_weights_iterations,
            'splits': splits,
            'split_factor': split_factor,
            'max_bin_count': max_bin_count,
            'custom_bin_count': custom_bin_count,
            'max_retries_exp_mechanism': max_retries_exp_mechanism
        }