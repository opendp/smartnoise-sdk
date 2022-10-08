from snsynth.aggregate_seeded import (
    AggregateSeededSynthesizer,
    AccuracyMode,
    FabricationMode,
    AggregateSeededDataset,
)

import pandas as pd
import numpy as np
import json
import pytest

from snsynth.transform.table import NoTransformer

def gen_data_frame_with_schema(schema, n_records):
    return pd.DataFrame(
        {k: list(np.random.choice(schema[k], size=n_records)) for k in schema},
        columns=schema.keys(),
    )


def gen_data_frame(number_of_records_to_generate):
    return pd.concat(
        [
            gen_data_frame_with_schema(
                {
                    "H1": ["1", "2", ""],
                    "H2": ["1", "2", "3", ""],
                    "H3": ["1", "2", "3", "4", "5", ""],
                    "H4": ["0", "1"],
                    "H5": ["0", "1"],
                    "H6": ["0", "1"],
                    "H7": ["0", "1"],
                    "H8": ["0", "1"],
                    "H9": ["0", "1"],
                    "H10": ["0", "1"],
                },
                number_of_records_to_generate // 2,
            ),
            gen_data_frame_with_schema(
                {
                    "H1": ["1", "2", ""],
                    "H2": ["4", "5", "6", ""],
                    "H3": ["6", "7", "8", "9", "10", ""],
                    "H4": ["0", "1"],
                    "H5": ["0", "1"],
                    "H6": ["0", "1"],
                    "H7": ["0", "1"],
                    "H8": ["0", "1"],
                    "H9": ["0", "1"],
                    "H10": ["0", "1"],
                },
                number_of_records_to_generate // 2,
            ),
        ],
        ignore_index=True,
    )


class TestAggregateSeeded:
    def setup(self):
        self.sensitive_df = gen_data_frame(10000)

    def test_synth_creation_with_default_params(self):
        synth = AggregateSeededSynthesizer()
        params = json.loads(str(synth.parameters))

        assert params == {
            "reporting_length": 3,
            "epsilon": 4.0,
            "delta": None,
            "percentile_percentage": 99,
            "percentile_epsilon_proportion": 0.01,
            "sigma_proportions": [1.0, 0.5, 0.3333333333333333],
            "number_of_records_epsilon_proportion": 0.005,
            "threshold": {"type": "Adaptive", "valuesByLen": {"3": 1.0, "2": 1.0}},
            "empty_value": "",
            "use_synthetic_counts": False,
            "weight_selection_percentile": 95,
            "aggregate_counts_scale_factor": None,
        }

    def test_synth_creation_with_provided_params(self):
        synth = AggregateSeededSynthesizer(
            reporting_length=4,
            epsilon=0.5,
            delta=0.001,
            percentile_percentage=95,
            percentile_epsilon_proportion=0.06,
            accuracy_mode=AccuracyMode.prioritize_short_combinations(),
            number_of_records_epsilon_proportion=0.006,
            fabrication_mode=FabricationMode.minimize(),
            empty_value="empty",
            use_synthetic_counts=True,
            weight_selection_percentile=96,
            aggregate_counts_scale_factor=2.0,
        )
        params = json.loads(str(synth.parameters))

        assert params == {
            "reporting_length": 4,
            "epsilon": 0.5,
            "delta": 0.001,
            "percentile_percentage": 95,
            "percentile_epsilon_proportion": 0.06,
            "sigma_proportions": [0.25, 0.3333333333333333, 0.5, 1.0],
            "number_of_records_epsilon_proportion": 0.006,
            "threshold": {
                "type": "Adaptive",
                "valuesByLen": {"4": 0.01, "3": 0.01, "2": 0.01},
            },
            "empty_value": "empty",
            "use_synthetic_counts": True,
            "weight_selection_percentile": 96,
            "aggregate_counts_scale_factor": 2.0,
        }

    def test_fit_with_list(self):
        raw_data = [
            self.sensitive_df.columns.tolist(),
            *self.sensitive_df.values.tolist(),
        ]
        synth = AggregateSeededSynthesizer()
        synth.fit(raw_data, transformer=NoTransformer())
        assert isinstance(synth.sample(10), list)

    def test_fit_with_data_frame(self):
        synth = AggregateSeededSynthesizer()
        synth.fit(self.sensitive_df, transformer=NoTransformer())
        assert isinstance(synth.sample(10), pd.DataFrame)

    def test_fit_with_dataset(self):
        raw_data = [
            self.sensitive_df.columns.tolist(),
            *self.sensitive_df.values.tolist(),
        ]
        synth = AggregateSeededSynthesizer()
        synth.fit(AggregateSeededDataset(raw_data), transformer=NoTransformer())
        assert isinstance(synth.sample(10), list)

    def test_fit_with_invalid_data(self):
        synth = AggregateSeededSynthesizer()

        with pytest.raises(ValueError):
            synth.fit([["A", "B"], ("1", "2")], transformer=NoTransformer())

    def test_get_sensitive_aggregates(self):
        synth = AggregateSeededSynthesizer()

        with pytest.raises(RuntimeError):
            synth.get_sensitive_aggregates()

        synth.fit(self.sensitive_df)
        aggregates = synth.get_sensitive_aggregates(
            combination_delimiter=",", reporting_length=2
        )

        assert isinstance(aggregates, dict)

    def test_get_dp_aggregates(self):
        synth = AggregateSeededSynthesizer()

        with pytest.raises(RuntimeError):
            synth.get_dp_aggregates()

        synth.fit(self.sensitive_df)
        dp_aggregates = synth.get_dp_aggregates(combination_delimiter=",")

        assert isinstance(dp_aggregates, dict)

    def test_get_dp_number_of_records(self):
        synth = AggregateSeededSynthesizer()

        with pytest.raises(RuntimeError):
            synth.get_dp_number_of_records()

        synth.fit(self.sensitive_df)

        assert synth.get_dp_number_of_records() > 0

    def test_sample_with_all_columns(self):
        synth = AggregateSeededSynthesizer()

        synth.fit(self.sensitive_df, transformer=NoTransformer())
        synthetic_data = synth.sample(100)

        assert set(synthetic_data.columns) == set(self.sensitive_df.columns)

    def test_sample_with_selected_columns(self):
        synth = AggregateSeededSynthesizer()
        selected_columns = ["H1", "H2", "H4", "H5", "H7"]

        synth.fit(self.sensitive_df, use_columns=selected_columns, transformer=NoTransformer())
        synthetic_data = synth.sample(100)

        assert set(synthetic_data.columns) == set(selected_columns)

    def test_fit_sample_with_sensitive_zeros(self):
        synth = AggregateSeededSynthesizer()

        sensitive_df = self.sensitive_df.copy()
        sensitive_df["H4"] = "0"

        synthetic_data = synth.fit_sample(sensitive_df, transformer=NoTransformer())

        assert len(synthetic_data) == len(self.sensitive_df)
        assert "0" not in synthetic_data.values

        synthetic_data = synth.fit_sample(sensitive_df, sensitive_zeros=["H4"], transformer=NoTransformer())

        assert len(synthetic_data) == len(self.sensitive_df)
        assert "0" in synthetic_data["H4"].values
