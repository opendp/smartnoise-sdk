import datetime
import random
import uuid
import numpy as np
import pandas as pd
import pytest

from snsynth.transform import *
from snsynth.transform.table import TableTransformer

from snsynth.transform.type_map import TypeMap

pums_csv_path = "../datasets/PUMS_null.csv"
pums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/

uuids = [str(uuid.uuid4()) for _ in range(len(pums))]
pums['guid'] = uuids

uuids2 = [str(uuid.uuid4()).replace('-', '') for _ in range(len(pums))]
pums['guid2'] = uuids2

dates = [datetime.datetime.now() + random.randint(0, 1000) * datetime.timedelta(days=1) for _ in range(len(pums))]
dates = [d.isoformat() for d in dates]
pums['when'] = dates

def rand_char():
    return chr(random.randint(ord('a'), ord('z')))
def rand_digit():
    return random.randint(0, 9)
def make_email():
    return f"{rand_char()}{rand_char()}{rand_char()}{rand_digit()}@{rand_char()}{rand_char()}{rand_char()}.com"
emails = [make_email() for _ in range(len(pums))]
pums['email'] = emails

def make_ssn():
    return f"{rand_digit()}{rand_digit()}{rand_digit()}-{rand_digit()}{rand_digit()}-{rand_digit()}{rand_digit()}{rand_digit()}{rand_digit()}"
ssns = [make_ssn() for _ in range(len(pums))]
pums['ssn'] = ssns

class TestTypeMap:
    def test_infer_pii(self):
        res = TypeMap.infer_column_types(pums)
        pii = dict(zip(res['columns'], res['pii']))
        assert(pii['guid'] == 'uuid4')
        assert(pii['guid2'] == 'uuid4')
        assert(pii['when'] == 'datetime')
        assert(pii['email'] == 'email')
        assert(pii['ssn'] == 'ssn')
        assert(pii['pid'] == 'sequence')
    def test_round_trip_table_infer(self):
        tt = TableTransformer.create(pums, style='cube')
        tt.fit(pums, epsilon=4.0)
        pums_encoded = tt.transform(pums)
        pums_decoded = tt.inverse_transform(pums_encoded)
        assert(list(pums.columns) == list(pums_decoded.columns))
        tt2 = TableTransformer.create(pums_decoded, style='cube')
        tt2.fit(pums_decoded, epsilon=4.0)
        assert(tt2.output_width == tt.output_width == 9)

    def _test_infer_exclude_each_column(self, data, columns):
        for col in columns:  # exclude each column once
            expected_columns = columns.copy()
            expected_columns.remove(col)

            res = TypeMap.infer_column_types(data, excluded_columns=[col])
            assert len(res["columns"]) == len(expected_columns)
            assert res["columns"] == expected_columns

    def test_infer_data_frame_exclude_column(self):
        columns = ["a", "b", "c"]
        df = pd.DataFrame(data=[[0, 8.15, "cat_1"]], columns=columns)

        self._test_infer_exclude_each_column(df, columns)

    def test_infer_tuples_and_ndarray_exclude_column(self):
        tuples = [(0, 8.15, "cat_1")]
        ndarray = np.array(tuples)
        columns = list(range(len(tuples[0])))

        self._test_infer_exclude_each_column(tuples, columns)
        self._test_infer_exclude_each_column(ndarray, columns)

    def test_get_transformers_with_list_constraints(self):
        columns = ["a", "b", "c"]
        constraints = {"a": "categorical", "b": "ordinal", "c": "continuous"}

        transformers = TypeMap.get_transformers(columns, constraints=constraints)
        assert len(transformers) == len(columns)

    def test_get_transformers_with_custom_constraints(self):
        columns = ["a", "b", "c", "d", "e"]
        anon_instance = AnonymizationTransformer(lambda: None)
        constraints = {
            "a": anon_instance,
            "b": LogTransformer,
            "c": "drop",
            "d": "LogTransformer",
            "e": "address",
        }

        transformers = TypeMap.get_transformers(columns, constraints=constraints)
        assert len(transformers) == len(columns)
        assert transformers[0] == anon_instance
        assert isinstance(transformers[1], LogTransformer)
        assert isinstance(transformers[2], DropTransformer)
        assert isinstance(transformers[3], LogTransformer)
        assert isinstance(transformers[4], AnonymizationTransformer)

    def test_get_transformers_with_invalid_constraint(self):
        with pytest.raises(ValueError, match="constraint.*?invalid"):
            TypeMap.get_transformers([0], constraints={0: None})

    def test_column_not_specified(self):
        with pytest.raises(ValueError, match="Column.*?not specified"):
            TypeMap.get_transformers([0])
