import pickle
import warnings

from snsynth.pytorch.nn import DPCTGAN

class DPSD:
    """Automated generative modeling and sampling tool.
    
    Based heavily on Synthetic Data Vault (SDV) (https://github.com/sdv-dev/SDV)
    
    Allows a user to generate differentially private synthetic data.
    Args:
        model (type):
            Class of the model to use. Defaults to ``snsynth.pytorch.nn.DPCTGAN``.
        model_kwargs (dict):
            Keyword arguments to pass to the model.
    """

    _model_instance = None
    DEFAULT_MODEL = DPCTGAN
    DEFAULT_MODEL_KWARGS = {
        'model': DPCTGAN,
        'model_kwargs': {
            # TODO
        }
    }

    def __init__(self, model=None, model_kwargs=None):
        if model is None:
            model = model or self.DEFAULT_MODEL
            if model_kwargs is None:
                model_kwargs = self.DEFAULT_MODEL_KWARGS

        self._model = model
        self._model_kwargs = (model_kwargs or dict()).copy()

    def fit(self, data, metadata):
        """Fit this DPSD instance to the dataset data.
        Args:
            data (Pandas DataFrame):
                Data to fit model to
            metadata (dict, str or Metadata):
                Metadata dict, path to the metadata JSON file or Metadata instance itself.
        """
        self._model_instance = self._model(metadata, **self._model_kwargs)
        self._model_instance.fit(data)

    def sample(self, num_rows=None):
        """Generate differentially private synthetic data for one table or the entire dataset.
        
        Args:
            num_rows (int):
                Amount of rows to sample. If ``None``, sample the same number of rows
                as there were in the original table.
        Returns:
            pandas.DataFrame:
                Returns a ``pandas.DataFrame``
        """
        if self._model_instance is None:
            raise ValueError('SDV instance has not been fitted')

        return self._model_instance.sample(
            num_rows
        )

    @classmethod
    def save(self, path):
        """Save this DPSD instance to the given path using pickle.
        Args:
            path (str):
                Path where the DPSD instance will be serialized.
        """
        with open(path, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path):
        """Load a DPSD instance from a given path.
        Args:
            path (str):
                Path from which to load the DPSD instance.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)