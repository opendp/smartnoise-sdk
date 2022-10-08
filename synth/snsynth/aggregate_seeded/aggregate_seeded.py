from pacsynth import (
    DpAggregateSeededSynthesizer,
    DpAggregateSeededParametersBuilder,
    AccuracyMode,
    FabricationMode,
)
from pacsynth import Dataset as AggregateSeededDataset
from snsynth.base import SDGYMBaseSynthesizer, Synthesizer
from functools import wraps

import pandas as pd

from snsynth.transform.table import TableTransformer


"""
Wrapper for Aggregate Seeded Synthesizer from pac-synth:
https://pypi.org/project/pac-synth/.

A differentially-private synthesizer that relies on DP Marginals to
build synthetic data. It will compute DP Marginals (called aggregates)
for your dataset up to and including a specified reporting length, and
synthesize data based on the computed aggregated counts.

For documentation please refer to:
    - https://github.com/microsoft/synthetic-data-showcase
    - https://github.com/microsoft/synthetic-data-showcase/tree/main/docs/dp
"""


class AggregateSeededSynthesizer(Synthesizer):
    """
    SmartNoise class wrapper for Aggregate Seeded Synthesizer from pac-synth.
    Works with Pandas data frames, raw data and follows norms set by other SmartNoise synthesizers.

    Reuses code and modifies it lightly from
    https://github.com/microsoft/synthetic-data-showcase/tree/main/packages/lib-pacsynth
    to achieve this.
    """

    def __init__(
        self,
        reporting_length=3,
        epsilon=4.0,
        delta=None,
        percentile_percentage=99,
        percentile_epsilon_proportion=0.01,
        accuracy_mode=AccuracyMode.prioritize_long_combinations(),
        number_of_records_epsilon_proportion=0.005,
        fabrication_mode=FabricationMode.uncontrolled(),
        empty_value="",
        use_synthetic_counts=False,
        weight_selection_percentile=95,
        aggregate_counts_scale_factor=None,
    ):
        """
        Wrapper for Aggregate Seeded Synthesizer from pac-synth.

        For more information about the parameters run `help('pacsynth.DpAggregateSeededParametersBuilder')`.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.reporting_length = reporting_length
        self.percentile_percentage = percentile_percentage
        self.percentile_epsilon_proportion = percentile_epsilon_proportion
        self.accuracy_mode = accuracy_mode
        self.number_of_records_epsilon_proportion = number_of_records_epsilon_proportion
        self.fabrication_mode = fabrication_mode
        self.empty_value = empty_value
        self.use_synthetic_counts = use_synthetic_counts
        self.weight_selection_percentile = weight_selection_percentile
        self.aggregate_counts_scale_factor = aggregate_counts_scale_factor
        self.synth = None

    def build_synthesizer(self):
        builder = (
            DpAggregateSeededParametersBuilder()
            .reporting_length(self.reporting_length)
            .epsilon(self.epsilon)
            .percentile_percentage(self.percentile_percentage)
            .percentile_epsilon_proportion(self.percentile_epsilon_proportion)
            .accuracy_mode(self.accuracy_mode)
            .number_of_records_epsilon_proportion(self.number_of_records_epsilon_proportion)
            .fabrication_mode(self.fabrication_mode)
            .empty_value(self.empty_value)
            .use_synthetic_counts(self.use_synthetic_counts)
            .weight_selection_percentile(self.weight_selection_percentile)
        )

        if self.aggregate_counts_scale_factor is not None:
            builder = builder.aggregate_counts_scale_factor(
                self.aggregate_counts_scale_factor
            )

        if self.delta is not None:
            builder = builder.delta(self.delta)

        self.reporting_length = self.reporting_length
        self.parameters = builder.build()
        self.synth = DpAggregateSeededSynthesizer(self.parameters)
        self.dataset = None
        self.pandas = False

    def fit(
        self, 
        data,
        *ignore,
        use_columns=None,
        sensitive_zeros=None,
        transformer=None,
        categorical_columns=None,
        ordinal_columns=None,
        continuous_columns=None,
        preprocessor_eps=0.0,
        nullable=False
        ):
        """
        Fit the synthesizer model on the data.

        This will compute the differently private aggregates used to
        synthesize data.

        All the columns are supposed to be categorical, non-categorical columns
        should be binned in advance.

        For more information run `help('pacsynth.Dataset')` and
        `help('pacsynth.DpAggregateSeededSynthesizer.fit')`.

        :param data: The data for fitting the synthesizer model.
        :type data: pd.DataFrame, list[list[str]], AggregateSeededDataset
        :param use_columns: List of column names to be used, defaults to None, meaning use all columns
        :type use_columns: list[str], optional
        :param sensitive_zeros: List of column names containing '0' that should not be turned into empty strings.
        :type sensitive_zeros: list[str], optional
        """
        train_data = self._get_train_data(
            data,
            style='cube',
            transformer=transformer,
            categorical_columns=categorical_columns, 
            ordinal_columns=ordinal_columns, 
            continuous_columns=continuous_columns, 
            nullable=True,
            preprocessor_eps=preprocessor_eps
        )

        if self._transformer is None:
            raise ValueError("We weren't able to fit a transformer to the data. Please check your data and try again.")

        if self.synth is None:
            self.build_synthesizer()

        if self._transformer.output_width > 0:
            colnames = ["column_{}".format(i) for i in range(len(train_data[0]))]
            data = [colnames] + [[str(v) for v in row] for row in train_data]

        if isinstance(data, list) and all(map(lambda row: isinstance(row, list), data)):
            self.dataset = AggregateSeededDataset(
                data, use_columns=use_columns, sensitive_zeros=colnames
            )
            self.pandas = False
        elif isinstance(data, pd.DataFrame):
            self.dataset = AggregateSeededDataset.from_data_frame(
                data, use_columns=use_columns, sensitive_zeros=sensitive_zeros
            )
            self.pandas = True
        elif isinstance(data, AggregateSeededDataset):
            self.dataset = data
            self.pandas = False
        else:
            raise ValueError(
                "data should be either in raw format (List[List[]]) or a be pandas data frame (pd.DataFrame)"
            )

        self.synth.fit(self.dataset)

    @wraps(SDGYMBaseSynthesizer.sample)
    def sample(self, samples=None):
        """
        Sample from the synthesizer model.

        This will sample records from the generated differentially private aggregates.

        If `samples` is `None`, the synthesizer will use all the available differentially
        private attributes counts to synthesize records
        (which will produce a number close to original number of records).

        For more information run `help('pacsynth.DpAggregateSeededSynthesizer.sample')`.

        :param samples: The number of samples to create
        :type samples: int, None
        :return: Generated data samples, the output type adjusts accordingly to the input data.
        :rtype: Dataframe, list[list[str]]
        """
        result = self.synth.sample(samples)

        if self._transformer is not None and self._transformer.output_width > 0:
            result = [[int(v) if v != '' else None for v in row] for row in result[1:]]
            result = self._transformer.inverse_transform(result)
            return result

        if self.pandas is True:
            result = AggregateSeededDataset.raw_data_to_data_frame(result)

        return result


    def get_sensitive_aggregates(
        self, combination_delimiter=";", reporting_length=None
        ):
        """
        Returns the aggregates for the sensitive dataset. For more information run `help('pacsynth.Dataset.get_aggregates')`.

        :param combination_delimiter: Combination delimiter to use, default to ';'
        :type combination_delimiter: str, optional
        :param reporting_length: Maximum length (inclusive) to compute attribute combinations for, defaults to the configured value in the synthesizer
        :type reporting_length: int, optional

        :return: A dictionary with the combination string representation as key and the combination count as value
        :rtype: dict[str, int]
        """
        if self.dataset is None:
            raise RuntimeError(
                "make sure 'fit' method has been successfully called first"
            )

        if reporting_length is None:
            reporting_length = self.reporting_length

        return self.dataset.get_aggregates(reporting_length, combination_delimiter)

    def get_dp_aggregates(self, combination_delimiter=";"):
        """
        Returns the aggregates for the sensitive dataset protected with differential privacy.

        For more information run `help('pacsynth.DpAggregateSeededSynthesizer.get_dp_aggregates')`.

        :param combination_delimiter: Combination delimiter to use, default to ';'
        :type combination_delimiter: str, optional
        :return: A dictionary with the combination string representation as key and the combination count as value
        :rtype: dict[str, int]
        """
        return self.synth.get_dp_aggregates(combination_delimiter)

    def get_dp_number_of_records(self):
        """
        Gets the differentially private number of records computed with the `.fit` method.

        This is different than the number of records specified in the sample method, synthesized
        in the synthetic data. This refers to the differentially private protected number of records
        in original sensitive dataset (Laplacian noise added).

        For more information run `help('pacsynth.DpAggregateSeededSynthesizer.get_dp_number_of_records')`.

        :return: Number of sensitive records protect with differential privacy
        :rtype: int
        """
        return self.synth.get_dp_number_of_records()
