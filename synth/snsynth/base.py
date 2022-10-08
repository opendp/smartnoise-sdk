import pandas as pd
import numpy as np

from snsynth.transform.table import TableTransformer

class SDGYMBaseSynthesizer:
    def fit(
        self, 
        data, *ignore, 
        transformer=None, 
        categorical_columns=[], 
        ordinal_columns=[], 
        continuous_columns=[],
        preprocessor_eps=0.0,
        nullable=False
        ):
        """
        Fit the synthesizer model on the data.

        :param data: The data for fitting the synthesizer model.
        :type data: pd.DataFrame, np.ndarray, or list of tuples
        :param transformer: The transformer to use to transform the data, defaults to None
        :type transformer: snsynth.transform.TableTransformer, optional
        :param categorical_columns: List of column names for categorical columns, defaults to None
        :type categorical_columns: list[str], optional
        :param ordinal_columns: List of column names for ordinal columns, defaults to None
        :type ordinal_columns: list[str], optional
        :param ordinal_columns: List of column names for ordinal columns, defaults to None
        :type ordinal_columns: list[str], optional
        :return: Data set containing the generated data samples.
        :rtype: pd.DataFrame, np.ndarray, or list of tuples
        """
        raise NotImplementedError

    def sample(self, n_samples):
        """
        Sample from the synthesizer model.

        :param n_samples: The number of samples to create
        :type samples: int
        :return: Data set containing the generated data samples.
        :rtype: pd.DataFrame, np.ndarray, or list of tuples
        """
        raise NotImplementedError

    def fit_sample(
        self, 
        data, *ignore, 
        transformer=None, 
        categorical_columns=[], 
        ordinal_columns=[], 
        continuous_columns=[],
        preprocessor_eps=0.0,
        nullable=False
        ):
        """
        Fit the synthesizer model and then generate a synthetic dataset of the same
        size of the input data.

        :param data: The data for fitting the synthesizer model.
        :type data: pd.DataFrame, np.ndarray, or list of tuples
        :param transformer: The transformer to use to transform the data, defaults to None
        :type transformer: snsynth.transform.TableTransformer, optional
        :param categorical_columns: List of column names for categorical columns, defaults to None
        :type categorical_columns: list[str], optional
        :param ordinal_columns: List of column names for ordinal columns, defaults to None
        :type ordinal_columns: list[str], optional
        :param ordinal_columns: List of column names for ordinal columns, defaults to None
        :type ordinal_columns: list[str], optional
        :return: Data set containing the generated data samples.
        :rtype: pd.DataFrame, np.ndarray, or list of tuples
        """
        self.fit(
            data, 
            transformer=transformer, 
            categorical_columns=categorical_columns, 
            ordinal_columns=ordinal_columns, 
            continuous_columns=continuous_columns,
            preprocessor_eps=preprocessor_eps,
            nullable=nullable,
        )
        if isinstance(data, pd.DataFrame):
            return self.sample(len(data))
        elif isinstance(data, np.ndarray):
            return self.sample(data.shape[0])
        elif isinstance(data, list):
            return self.sample(len(data))
        else:
            raise TypeError('Data must be a pandas DataFrame, numpy array, or list of tuples')

synth_map = {
    'mwem': {
        'class': 'snsynth.mwem.MWEMSynthesizer'
    },
    'dpctgan' : {
        'class': 'snsynth.pytorch.nn.dpctgan.DPCTGAN'
    },
    'patectgan' : {
        'class': 'snsynth.pytorch.nn.patectgan.PATECTGAN'
    },
    'mst': {
        'class': 'snsynth.mst.mst.MSTSynthesizer'
    },
    'pacsynth': {
        'class': 'snsynth.aggregate_seeded.AggregateSeededSynthesizer'
    },
    'dpgan': {
        'class': 'snsynth.pytorch.nn.dpgan.DPGAN'
    },
    'pategan': {
        'class': 'snsynth.pytorch.nn.pategan.PATEGAN'
    }
}

class Synthesizer(SDGYMBaseSynthesizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _get_train_data(self, data, *ignore, style, transformer, categorical_columns, ordinal_columns, continuous_columns, nullable, preprocessor_eps):
        if transformer is None:
            self._transformer = TableTransformer.create(data, style=style,
                categorical_columns=categorical_columns,
                continuous_columns=continuous_columns,
                ordinal_columns=ordinal_columns,
                nullable=nullable,)
        elif isinstance(transformer, TableTransformer):
            self._transformer = transformer
        else:
            raise ValueError("transformer must be a TableTransformer object or None.  See the updated documentation.")
        if not self._transformer.fit_complete:
            if self._transformer.needs_epsilon and (preprocessor_eps is None or preprocessor_eps == 0.0):
                raise ValueError("Transformer needs some epsilon to infer bounds.  If you know the bounds, pass them in to save budget.  Otherwise, set preprocessor_eps to a value > 0.0 and less than the training epsilon.  Preprocessing budget will be subtracted from training budget.")
            self._transformer.fit(
                data,
                epsilon=preprocessor_eps
            )
            eps_spent, _ = self._transformer.odometer.spent
            if eps_spent > 0.0:
                self.epsilon -= eps_spent
                print(f"Spent {eps_spent} epsilon on preprocessor, leaving {self.epsilon} for training")
                if self.epsilon < 10E-3:
                    raise ValueError("Epsilon remaining is too small!")
        train_data = self._transformer.transform(data)
        return train_data
    # factory method
    @classmethod
    def create(cls, synth, *args, **kwargs):
        synth = synth.lower()
        if synth not in synth_map:
            raise ValueError('Synthesizer {} not found'.format(synth))
        synth_class = synth_map[synth]['class']
        synth_module, synth_class = synth_class.rsplit('.', 1)
        synth_module = __import__(synth_module, fromlist=[synth_class])
        synth_class = getattr(synth_module, synth_class)
        return synth_class(*args, **kwargs)

