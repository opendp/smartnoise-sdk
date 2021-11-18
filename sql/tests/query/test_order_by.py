from snsql import *
import pandas as pd
import numpy as np

from snsql.metadata import Metadata


query = "SELECT start_date, SUM(sales), start_time FROM PUMS.PUMS GROUP BY start_date, start_time ORDER BY start_date DESC, start_time"

pre_aggregated = [
    ('keycount', 'start_date', 'start_time', 'sum_sales'),
    (1000, '2018-01-03', '16:00:03', 10000000.0),
    (1000, '2018-01-03', '15:10:04', 10000000.0),
    (1000, '2018-02-03', '12:20:05', 10000000.0),
    (1000, '2018-02-03', '11:30:06', 10000000.0),
    (1000, '2018-02-03', '10:40:07', 10000000.0),
    (1000, '2018-02-02', '09:50:08', 50000000.0),
    (1000, '2018-02-02', '08:10:09', 20000000.0),
    (1000, '2018-01-02', '07:20:10', 20000000.0),
    (1000, '2018-01-01', '06:30:11', 30000000.0)
]

privacy = Privacy(epsilon=3.0, delta=0.1)

metadata = Metadata.from_dict({
    "": {
        "PUMS": {
            "PUMS":{
                "censor_dims": True,
                "row_privacy": False,
                "uuid": {
                    "type": "int",
                    "private_id": True
                },
                "start_date": {
                    "type": "datetime"
                },
                "start_time": {
                    "type": "string"
                },
                "sales": {
                    "type": "float",
                    "upper": 10000,
                    "lower": 0.0
                }
            }
        }
    }
})


reader = from_connection(None, engine="sqlserver", metadata=metadata, privacy=Privacy(epsilon=1.0, delta=10e-6))

class TestOrderBy:
    def test_order_by_list(self, test_databases):
        res = reader.execute_df(query, pre_aggregated=pre_aggregated)
        assert(str(res['start_date'][0]) == '2018-02-03')
        assert(res['start_time'][0] == '10:40:07')
    def test_order_by_spark(self, test_databases):
        priv = test_databases.get_private_reader(
            privacy=privacy,
            database="PUMS_pid",
            engine="spark"
        )
        if priv:
            pre_agg = priv.reader.api.createDataFrame(pre_aggregated[1:], pre_aggregated[0])
            res = reader.execute_df(query, pre_aggregated=pre_agg)
            assert(str(res['start_date'][0]) == '2018-02-03')
            assert(res['start_time'][0] == '10:40:07')

