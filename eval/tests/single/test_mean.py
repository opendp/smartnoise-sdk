
class TestMean:
    def test_mean_pums(self, pums_df):
        count = pums_df.count()
        assert count == 1000
    def test_mean_parquet(self, pums_agg_df):
        count = pums_agg_df.count()
        assert count > 10000
