from sneval import Metric

class TestMedian:
    def test_median_pums(self, pums_dataset):
        median = Metric.create("Median", column_name="income")
        result = median.compute(pums_dataset)
        assert result > 17000
        assert result < 20000
    
    def test_median_pums_large(self, pums_large_dataset):
        median = Metric.create("Median", column_name="income")
        result = median.compute(pums_large_dataset)
        assert result > 17000
        assert result < 20000
