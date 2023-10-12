from sneval import Metric

class TestMeanProportionalErrorInCount:
    def test_mean_prop_error_in_count(self, test_all_plus_last_6k_dataset, test_all_minus_last_12k_dataset):
        edges = [1, 10, 100, 1000, 10000, 100000]
        mean_abs_error_in_count = Metric.create("MeanProportionalErrorInCount", categorical_columns=["ProductID","CustomerRegion"], edges=edges)
        result = mean_abs_error_in_count.compute(test_all_plus_last_6k_dataset, test_all_minus_last_12k_dataset)
        value = result['value']

        for i in range(1, len(edges)):
            bin_value = value[f'Bin {i}']
            if bin_value != 'NA':  # We only check if the value is not 'NA'
                lower_bound = 0.5 * 1.5  # Expect a ~1.5% error with a torlerance
                upper_bound = 1.5 * 1.5
                assert lower_bound <= bin_value <= upper_bound, f"Value in Bin {i} is out of bounds!"