import datetime
import random
import uuid
import pandas as pd
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
