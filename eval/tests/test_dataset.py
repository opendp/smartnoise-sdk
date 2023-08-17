class TestDataset:
    def test_pums(self, pums_dataset):
        count = pums_dataset.source.count()
        assert count == 1000
        assert pums_dataset.is_aggregated == False
        assert pums_dataset.is_row_privacy == True
    def test_pums_agg(self, pums_agg_dataset):
        count = pums_agg_dataset.source.count()
        assert count > 10000
        assert pums_agg_dataset.is_aggregated == True
    def test_pums_id(self, pums_id_dataset):
        count = pums_id_dataset.source.count()
        assert count > 1500
        assert pums_id_dataset.is_row_privacy == False
    
    def test_aggregate_pums(self, pums_dataset):
        pums_agg = pums_dataset.aggregate()
        assert pums_agg.is_aggregated == True
        assert pums_agg.is_row_privacy == False
        assert pums_agg.source.count() < 1000

    def test_aggregate_pums_agg(self, pums_agg_dataset):
        pums_agg = pums_agg_dataset.aggregate()
        assert pums_agg.is_aggregated == True
        assert pums_agg.is_row_privacy == False
        assert pums_agg.source.count() == pums_agg_dataset.source.count()
        assert pums_agg.matches(pums_agg_dataset)
        assert pums_agg_dataset.matches(pums_agg)

    def test_compare_two_agg(self, pums_dataset, pums_id_dataset):
        pums_agg = pums_dataset.aggregate()
        pums_id_agg = pums_id_dataset.aggregate()
        assert pums_agg.matches(pums_id_agg)
        assert pums_id_agg.matches(pums_agg)
