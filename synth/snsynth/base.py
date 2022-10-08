import pandas as pd
import numpy as np

from snsynth.transform.table import TableTransformer

class SDGYMBaseSynthesizer:
    def fit(
        self, 
        data, 
        *ignore, 
        transformer=None, 
        categorical_columns=[], 
        ordinal_columns=[], 
        continuous_columns=[],
        preprocessor_eps=0.0,
        nullable=False
        ):
        """
        Fit the synthesizer model on the data.

        :param data: The private data used to fit the synthesizer.
        :type data: pd.DataFrame, np.ndarray, or list of tuples
        :param transformer: The transformer to use to preprocess the data.  If no transformer 
            is provided, the synthesizer will attempt to choose a transformer suitable for that 
            synthesizer.  To prevent the synthesizer from choosing a transformer, pass in
            snsynth.transform.NoTransformer().
        :type transformer: snsynth.transform.TableTransformer, optional
        :param categorical_columns: List of column names or indixes to be treated as categorical columns, used as hints when no transformer is provided.
        :type categorical_columns: list[], optional
        :param ordinal_columns: List of column names or indices to be treated as ordinal columns, used as hints when no transformer is provided.
        :type ordinal_columns: list[], optional
        :param ordinal_columns: List of column names or indices to be treated as ordinal columns, used as hints when no transformer is provided.
        :type ordinal_columns: list[], optional
        :param preprocessor_eps: The epsilon value to use when preprocessing the data.  This epsilon budget is subtracted from the
            budget supplied when creating the synthesizer, but is only used if the preprocessing requires
            privacy budget, for example if bounds need to be inferred for continuous columns.  This value defaults to
            0.0, and the synthesizer will raise an error if the budget is not sufficient to preprocess the data.
        :type preprocessor_eps: float, optional
        :param nullable: Whether or not to allow null values in the data.  This is only used if no transformer is provided,
            and is used as a hint when inferring transformers.
        """
        raise NotImplementedError

    def sample(self, n_rows):
        """
        Sample rows from the synthesizer.

        :param n_rows: The number of rows to create
        :type n_rows: int
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
        nullable=False,
        **kwargs
        ):
        """
        Fit the synthesizer model and then generate a synthetic dataset of the same
        size of the input data.

        :param data: The private data used to fit the synthesizer.
        :type data: pd.DataFrame, np.ndarray, or list of tuples
        :param transformer: The transformer to use to preprocess the data.  If no transformer 
            is provided, the synthesizer will attempt to choose a transformer suitable for that 
            synthesizer.  To prevent the synthesizer from choosing a transformer, pass in
            snsynth.transform.NoTransformer().
        :type transformer: snsynth.transform.TableTransformer, optional
        :param categorical_columns: List of column names or indixes to be treated as categorical columns, used as hints when no transformer is provided.
        :type categorical_columns: list[], optional
        :param ordinal_columns: List of column names or indices to be treated as ordinal columns, used as hints when no transformer is provided.
        :type ordinal_columns: list[], optional
        :param ordinal_columns: List of column names or indices to be treated as ordinal columns, used as hints when no transformer is provided.
        :type ordinal_columns: list[], optional
        :param preprocessor_eps: The epsilon value to use when preprocessing the data.  This epsilon budget is subtracted from the
            budget supplied when creating the synthesizer, but is only used if the preprocessing requires
            privacy budget, for example if bounds need to be inferred for continuous columns.  This value defaults to
            0.0, and the synthesizer will raise an error if the budget is not sufficient to preprocess the data.
        :type preprocessor_eps: float, optional
        :param nullable: Whether or not to allow null values in the data.  This is only used if no transformer is provided,
            and is used as a hint when inferring transformers.
        """
        self.fit(
            data, 
            transformer=transformer, 
            categorical_columns=categorical_columns, 
            ordinal_columns=ordinal_columns, 
            continuous_columns=continuous_columns,
            preprocessor_eps=preprocessor_eps,
            nullable=nullable,
            **kwargs
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
    @classmethod
    def list_synthesizers(cls):
        """
        List the available synthesizers.
        
        :return: List of available synthesizer names.
        :rtype: list[str]
        """
        return list(synth_map.keys())
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
    def create(cls, synth=None, epsilon=None, *args, **kwargs):
        """
        Create a differentially private synthesizer.

        :param synth: The name of the synthesizer to create.  If called from an instance of a Synthesizer subclass, creates
            an instance of the specified synthesizer.  Allowed synthesizers are available from
            the list_synthesizers() method.
        :type synth: str or Synthesizer class, required
        :param epsilon: The privacy budget to be allocated to the synthesizer.  This budget will be
            used when the synthesizer is fit to the data.
        :type epsilon: float, required
        :param args: Positional arguments to pass to the synthesizer constructor.
        :type args: list, optional
        :param kwargs: Keyword arguments to pass to the synthesizer constructor.  At a minimum,
            the epsilon value must be provided.  Any other hyperparameters can be provided
            here.  See the documentation for each specific synthesizer for details about available
            hyperparameter.
        :type kwargs: dict, optional

        """
        if synth is None or (isinstance(synth, type) and issubclass(synth, Synthesizer)):
            clsname = cls.__module__ + '.' + cls.__name__ if synth is None else synth.__module__ + '.' + synth.__name__
            if clsname == 'snsynth.base.Synthesizer':
                raise ValueError("Must specify a synthesizer to use.")
            matching_keys = [k for k, v in synth_map.items() if v['class'] == clsname]
            if len(matching_keys) == 0:
                raise ValueError(f"Synthesizer {clsname} not found in map.")
            elif len(matching_keys) > 1:
                raise ValueError(f"Synthesizer {clsname} found multiple times in map.")
            else:
                synth = matching_keys[0]
        if isinstance(synth, str):
            synth = synth.lower()
            if synth not in synth_map:
                raise ValueError('Synthesizer {} not found'.format(synth))
            synth_class = synth_map[synth]['class']
            synth_module, synth_class = synth_class.rsplit('.', 1)
            synth_module = __import__(synth_module, fromlist=[synth_class])
            synth_class = getattr(synth_module, synth_class)
            return synth_class(epsilon=epsilon, *args, **kwargs)
        else:
            raise ValueError('Synthesizer must be a string or a class')