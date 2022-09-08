from pacsynth import DpAggregateSeededSynthesizer, DpAggregateSeededParametersBuilder, AccuracyMode, FabricationMode, Dataset
from snsynth.base import SDGYMBaseSynthesizer
from snsynth.aggregate_seeded.accuracy_mode import AccuracyMode
from snsynth.aggregate_seeded.fabrication_mode import FabricationMode
from snsynth.aggregate_seeded.dataset import AggregateSeededDataset
from functools import wraps

import pandas as pd


"""
Wrapper for Aggregate Seeded synthesizer from pac-synth:
https://pypi.org/project/pac-synth/.

A differentially-private synthesizer that relies on DP Marginals to
build synthetic data. It will compute DP Marginals (called aggregates)
for your dataset up to and including a specified reporting length, and
synthesize data based on the computed aggregated counts.

For documentation please refer to:
    - https://github.com/microsoft/synthetic-data-showcase
    - https://github.com/microsoft/synthetic-data-showcase/tree/main/docs/dp
"""


class AggregateSeededSynthesizer(SDGYMBaseSynthesizer):
    """
    SmartNoise class wrapper for Aggregate Seeded synthesizer from pac-synth.
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
        builder = DpAggregateSeededParametersBuilder() \
            .reporting_length(reporting_length) \
            .epsilon(epsilon) \
            .percentile_percentage(percentile_percentage) \
            .percentile_epsilon_proportion(percentile_epsilon_proportion) \
            .accuracy_mode(accuracy_mode) \
            .number_of_records_epsilon_proportion(number_of_records_epsilon_proportion) \
            .fabrication_mode(fabrication_mode) \
            .empty_value(empty_value) \
            .use_synthetic_counts(use_synthetic_counts) \
            .weight_selection_percentile(weight_selection_percentile)

        if aggregate_counts_scale_factor != None:
            builder = builder.aggregate_counts_scale_factor(aggregate_counts_scale_factor)

        if delta != None:
            builder = builder.delta(delta)

        self.reporting_length = reporting_length
        self.parameters = builder.build()
        self.synth = DpAggregateSeededSynthesizer(
            self.parameters
        )
        self.dataset = None
        self.pandas = False

    @wraps(SDGYMBaseSynthesizer.fit)
    def fit(self, data, categorical_columns=None, ordinal_columns=None, sensitive_zeros=None):
        assert ordinal_columns == None, 'ordinal columns should be binned and transformed in categorical columns'

        if isinstance(data, list) and all(map(lambda row: isinstance(row, list), data)):
            self.dataset = AggregateSeededDataset(
                data,
                use_columns=categorical_columns,
                sensitive_zeros=sensitive_zeros
            )
            self.pandas = False
        elif isinstance(data, pd.DataFrame):
            self.dataset = AggregateSeededDataset.from_data_frame(
                data,
                use_columns=categorical_columns,
                sensitive_zeros=sensitive_zeros
            )
            self.pandas = True
        elif isinstance(data, AggregateSeededDataset):
            self.dataset = data
            self.pandas = False
        else:
            raise ValueError(
                'data should be either in raw format (List[List[]]) or a be pandas data frame (pd.DataFrame)'
            )

        self.synth.fit(self.dataset)

    @wraps(SDGYMBaseSynthesizer.sample)
    def sample(self, samples=None):
        result = self.synth.sample(samples)

        if self.pandas == True:
            result = AggregateSeededDataset.raw_data_to_data_frame(result)

        return result

    def get_sensitive_aggregates(self, combination_delimiter=';', reporting_length=None):
        if self.dataset == None:
            raise RuntimeError("make sure 'fit' method has been successfully called first")

        if reporting_length == None:
            reporting_length = self.reporting_length

        return self.dataset.get_aggregates(
            reporting_length, combination_delimiter
        )

    def get_dp_aggregates(self, combination_delimiter=';'):
        return self.synth.get_dp_aggregates(combination_delimiter)

    def get_dp_number_of_records(self):
        return self.synth.get_dp_number_of_records()
