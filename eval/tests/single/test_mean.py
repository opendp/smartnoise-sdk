from sneval import Metric

class TestMean:
    def test_mean_pums(self, pums_dataset):
        mean = Metric.create("Mean", column_name="income")
        result = mean.compute(pums_dataset)
        value = result['value']
        assert value > 31000
        assert value < 35000
    def test_mean_pums_pre_agg(self, pums_agg_dataset, pums_large_dataset):
        mean = Metric.create("Mean", column_name="income")
        result_1 = mean.compute(pums_agg_dataset)
        value_1 = result_1['value']
        result_2 = mean.compute(pums_large_dataset)
        value_2 = result_2['value']
        assert value_1 == value_2
