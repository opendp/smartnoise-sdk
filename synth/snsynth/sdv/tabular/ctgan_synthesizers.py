"""Wrapper around PATECTGAN and DPCTGAN models."""

import numpy as np
from snsynth.pytorch.nn import DPCTGAN, PATECTGAN
from snsynth.preprocessors import GeneralTransformer
from snsynth.pytorch import PytorchDPSynthesizer

from snsynth.sdv.tabular.base import BaseTabularModel

# NOTE: Adding this as optional import, may need adjusting
try: 
    from snsynth.sdv.metadata import Table
except ImportError: 
    Table = None

class SmartnoiseCTGANModel(BaseTabularModel):
    """Base class for all the CTGAN models.
    The ``CTGANModel`` class provides a wrapper for all the CTGAN models.
    """

    _MODEL_CLASS = None
    _model_kwargs = None

    _DTYPE_TRANSFORMERS = {
        'O': None
    }

    def _build_model(self):
        # TODO: Update with new Transformers
        inner_model = self._MODEL_CLASS(**self._model_kwargs)
        return PytorchDPSynthesizer(self._model_kwargs['epsilon'], inner_model, GeneralTransformer())

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

class PATECTGAN(SmartnoiseCTGANModel):
    _MODEL_CLASS = PATECTGAN

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 embedding_dim=128,
                 generator_dim=(256, 256),
                 discriminator_dim=(256, 256),
                 generator_lr=2e-4,
                 generator_decay=1e-6,
                 discriminator_lr=2e-4,
                 discriminator_decay=1e-6,
                 batch_size=500,
                 discriminator_steps=1,
                 log_frequency=False,
                 verbose=False,
                 epochs=300,
                 pac=1,
                 cuda=True,
                 epsilon=1,
                 binary=False,
                 regularization=None, #dragan
                 loss="cross_entropy",
                 teacher_iters=5,
                 student_iters=5,
                 sample_per_teacher=1000,
                 delta=None,
                 noise_multiplier=1e-3,
                 moments_order=100,
                 category_epsilon_pct=0.1,
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
            'embedding_dim': embedding_dim,
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'cuda': cuda,
            'epsilon': epsilon,
            'binary': binary,
            'regularization': regularization,
            'loss': loss,
            'teacher_iters': teacher_iters,
            'student_iters': student_iters,
            'sample_per_teacher': sample_per_teacher,
            'delta': delta,
            'noise_multiplier': noise_multiplier,
            'moments_order': moments_order,
            'category_epsilon_pct': category_epsilon_pct
        }

class DPCTGAN(SmartnoiseCTGANModel):
    _MODEL_CLASS = DPCTGAN

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 embedding_dim=128,
                 generator_dim=(256, 256),
                 discriminator_dim=(256, 256),
                 generator_lr=2e-4,
                 generator_decay=1e-6,
                 discriminator_lr=2e-4,
                 discriminator_decay=1e-6,
                 batch_size=500,
                 discriminator_steps=1,
                 log_frequency=False,
                 verbose=True,
                 epochs=300,
                 pac=1,
                 cuda=True,
                 disabled_dp=False,
                 delta=None,
                 sigma=5,
                 max_per_sample_grad_norm=1.0,
                 epsilon=1,
                 loss="cross_entropy",
                 category_epsilon_pct=0.1,
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
            'embedding_dim': embedding_dim,
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'cuda': cuda,
            'disabled_dp': disabled_dp,
            'delta': delta,
            'sigma': sigma,
            'max_per_sample_grad_norm': max_per_sample_grad_norm,
            'epsilon': epsilon,
            'loss': loss,
            'category_epsilon_pct': category_epsilon_pct
        }