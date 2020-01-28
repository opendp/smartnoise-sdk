# OSS Trusted analyst
import sklearn.datasets
import pandas as pd

from burdock.query.sql import DataFrameReader, MetadataLoader, execute_private_query

def test_sklearn_query():
    sklearn_dataset = sklearn.datasets.load_iris()
    sklearn_df = pd.DataFrame(data=sklearn_dataset.data, columns=sklearn_dataset.feature_names)

    schema_dict = {"Database": {
                     "dbo": {
                       "iris": {
                         "rows": 150,
                         "sepal length (cm)": {
                            "type": "float",
                            "lower": 4,
                            "upper": 8},
                         "sepal width (cm)": {
                            "type": "float",
                            "lower": 2,
                            "upper": 5},
                         "petal length (cm)": {
                            "type": "float",
                            "lower": 1,
                            "upper": 7},
                         "petal width (cm)": {
                            "type": "float",
                            "lower": 0,
                            "upper": 3}
                        }
                      }
                    }
                  }

    schema = MetadataLoader.from_dict(schema_dict)
    reader = DataFrameReader(schema, sklearn_df)
    rowset = execute_private_query(reader, schema, 0.3, 'SELECT AVG("petal width (cm)") FROM dbo.iris')
    df = pd.DataFrame(rowset[1:], columns=rowset[0])
    assert df is not None
    assert len(df) == 1
