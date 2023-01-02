import pandas as pd
import math
from snsynth.transform import *

pums = pd.read_csv('../datasets/PUMS_pid.csv', index_col=None)

class TestTableTransformWithMixedInference:
    def test_infer_only(self):
        # only inference
        tt = TableTransformer.create(pums, style='cube')
        tt.fit(pums, epsilon=1.0)
        print(tt.odometer.spent)
        income = tt.transformers[4]
        assert(income.fit_upper != 400_000)
        pid = tt.transformers[6]
        assert(isinstance(pid, AnonymizationTransformer))
        assert('SequenceCounter' in str(type(pid.fake)))
    def test_infer_with_declared(self):
        # inference with declared income
        tt = TableTransformer.create(
            pums, 
            style='cube',
            constraints={
                'income': BinTransformer(bins=20, lower=0, upper=400_000)
            }
        )
        tt.fit(pums, epsilon=1.0)
        assert(tt.odometer.spent[0] == 0.0)
        income = tt.transformers[4]
        assert(income.fit_upper == 400_000)
        pid = tt.transformers[6]
        assert(isinstance(pid, AnonymizationTransformer))
        assert('SequenceCounter' in str(type(pid.fake)))
    def test_declare_with_string(self):
        # declare using a string
        tt = TableTransformer.create(
            pums, 
            style='cube',
            constraints={
                'pid': "uuid4"
            }
        )
        tt.fit(pums, epsilon=1.0)
        assert(tt.odometer.spent[0] == 1.0)
        income = tt.transformers[4]
        assert(income.fit_upper != 400_000)
        pid = tt.transformers[6]
        assert(isinstance(pid, AnonymizationTransformer))
        assert('SequenceCounter' not in str(type(pid.fake)))
    def test_declare_with_mixed(self):
        # declare using a mix
        tt = TableTransformer.create(
            pums, 
            style='cube',
            constraints={
                'pid': 'uuid4',
                'income': 
                    ChainTransformer([
                        LogTransformer(),
                        BinTransformer(bins=20, lower=0, upper=math.log(400_000))
                    ])
            }
        )
        tt.fit(pums, epsilon=1.0)
        print(tt.odometer.spent)
        income = tt.transformers[4]
        assert(isinstance(income, ChainTransformer))
        pid = tt.transformers[6]
        assert(isinstance(pid, AnonymizationTransformer))
        assert('SequenceCounter' not in str(type(pid.fake)))
