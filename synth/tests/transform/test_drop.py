from snsynth.transform.drop import DropTransformer


class TestDrop:
    def _test_drop_all_indices(self, data):
        drop = DropTransformer()
        assert drop.fit_complete
        transformed = drop.transform(data)
        assert len(transformed) == len(data)
        assert all(v is None for v in transformed)
        inversed = drop.inverse_transform(transformed)
        assert len(inversed) == len(data)
        assert all(v is None for v in inversed)

    def test_list_of_values(self):
        str_list = ["a", "b", "c"]

        self._test_drop_all_indices(str_list)

    def test_tuples(self):
        tuples = [(0, "x"), (1, "x"), (2, "x")]

        self._test_drop_all_indices(tuples)

    def test_tuples_drop_single_index(self):
        tuples = [(0, "x"), (1, "x"), (2, "x")]

        drop = DropTransformer()
        assert drop.fit_complete
        transformed = drop.transform(tuples, idx=0)
        assert len(transformed) == len(tuples)
        assert all(v == "x" for v, in transformed)
        inversed = drop.inverse_transform(transformed, idx=0)
        assert len(inversed) == len(tuples)
        assert all(v == "x" for v, in inversed)
