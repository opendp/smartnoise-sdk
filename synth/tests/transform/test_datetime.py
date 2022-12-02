from snsynth.transform.datetime import DateTimeTransformer

class TestDateTimeTransformer:
    def test_dates(self):
        dates = ["2021-12-03", "2022-11-24", "2023-10-15", "2024-09-06", "2025-07-28", "2026-06-19", "2027-05-10", "2028-03-31", "2029-02-22", "2030-01-13"]
        dt = DateTimeTransformer()
        dt.fit(dates)
        dates_encoded = dt.transform(dates)
        dates_decoded = dt.inverse_transform(dates_encoded)
        assert(dt.format == "date")
        assert(dates == dates_decoded)
    def test_dates_alt_epoch(self):
        dates = ["2021-12-03", "2022-11-24", "2023-10-15", "2024-09-06", "2025-07-28", "2026-06-19", "2027-05-10", "2028-03-31", "2029-02-22", "2030-01-13"]
        dt = DateTimeTransformer(epoch="1900-01-21")
        dt.fit(dates)
        dates_encoded = dt.transform(dates)
        dates_decoded = dt.inverse_transform(dates_encoded)
        assert(dt.format == "date")
        assert(dates == dates_decoded)
    def test_datetimes(self):
        dates = ["2021-12-03T01:23:34", "2022-11-24T23:43:21", "2023-10-15T01:23:34", "2024-09-06T04:43:21", "2025-07-28T21:23:34", "2026-06-19T04:43:21", "2027-05-10T01:23:34", "2028-03-31T04:43:21", "2029-02-22T01:23:34", "2030-01-13T04:43:21"]
        dt = DateTimeTransformer()
        dt.fit(dates)
        dates_encoded = dt.transform(dates)
        dates_decoded = dt.inverse_transform(dates_encoded)
        assert(dt.format == "datetime")
        assert(dates == dates_decoded)
    def test_times(self):
        dates = ["01:23:34", "23:43:21", "01:23:34", "04:43:21", "21:23:34", "04:43:21", "01:23:34", "14:43:21", "11:23:34", "04:43:21.221010"]
        dt = DateTimeTransformer()
        dt.fit(dates)
        dates_encoded = dt.transform(dates)
        dates_decoded = dt.inverse_transform(dates_encoded)
        assert(dt.format == "time")
        assert(dates == dates_decoded)
    def test_times_alt_epoch(self):
        dates = ["01:23:34", "23:43:21", "01:23:34", "04:43:21", "21:23:34", "04:43:21", "01:23:34", "14:43:21", "11:23:34", "04:43:21.221010"]
        dt = DateTimeTransformer("1939-02-14")
        dt.fit(dates)
        dates_encoded = dt.transform(dates)
        dates_decoded = dt.inverse_transform(dates_encoded)
        assert(dt.format == "time")
        assert(dates == dates_decoded)
    def test_times_alt_epoch_2(self):
        dates = ["01:23:34", "23:43:21", "01:23:34", "04:43:21", "21:23:34", "04:43:21", "01:23:34", "14:43:21", "11:23:34", "04:43:21.221010"]
        dt = DateTimeTransformer("21:23:34")
        dt.fit(dates)
        dates_encoded = dt.transform(dates)
        dates_decoded = dt.inverse_transform(dates_encoded)
        assert(dt.format == "time")
        assert(dates == dates_decoded)
    def test_dates_negative(self):
        dates = ["1889-12-03", "1865-11-24", "1812-10-15", "1776-09-06", "1666-07-28", "1312-06-19", "1822-05-10", "1918-03-31", "1945-02-22", "1941-01-13"]
        dt = DateTimeTransformer()
        dt.fit(dates)
        dates_encoded = dt.transform(dates)
        dates_decoded = dt.inverse_transform(dates_encoded)
        assert(all([x < 0 for x in dates_encoded]))
        assert(dt.format == "date")
        assert(dates == dates_decoded)
    def test_times_alt_epoch_null(self):
        dates = ["01:23:34", "23:43:21", None, "04:43:21", "21:23:34", "04:43:21", "01:23:34", None, "11:23:34", "04:43:21.221010"]
        dt = DateTimeTransformer("1939-02-14")
        dt.fit(dates)
        dates_encoded = dt.transform(dates)
        dates_decoded = dt.inverse_transform(dates_encoded)
        assert(dt.format == "time")
        assert(dates == dates_decoded)
    def test_dates_alt_null(self):
        dates = [None, "2022-11-24", "2023-10-15", "2024-09-06", "2025-07-28", None, "2027-05-10", "2028-03-31", "2029-02-22", "2030-01-13"]
        dt = DateTimeTransformer(epoch="1900-01-21")
        dt.fit(dates)
        dates_encoded = dt.transform(dates)
        dates_decoded = dt.inverse_transform(dates_encoded)
        assert(dt.format == "date")
        assert(dates == dates_decoded)
