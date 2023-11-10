from sneval import Metric

class TestMeanAbsoluteErrorInCount:
    def test_mean_abs_error_in_count(self, test_all_plus_last_6k_dataset, test_all_minus_last_12k_dataset):
        edges = [1, 10, 100, 1000, 10000, 100000]
        mean_abs_error_in_count = Metric.create("MeanAbsoluteErrorInCount", categorical_columns=["ProductID","CustomerRegion"], edges=edges)
        result = mean_abs_error_in_count.compute(test_all_plus_last_6k_dataset, test_all_minus_last_12k_dataset)
        value = result['value']

        for i in range(1, len(edges)):
            bin_value = value[f'Bin {i}']
            if bin_value != 'NA':  # We only check if the value is not 'NA'
                half_bin_max_size = edges[i] / 2
                lower_bound = 0.5 * 0.015 * half_bin_max_size
                upper_bound = 1.5 * 0.015 * half_bin_max_size
                assert lower_bound <= bin_value <= upper_bound, f"Value in Bin {i} is out of bounds!"