# OSS Trusted analyst
import sklearn.datasets
import pandas as pd


from burdock.sql import execute_private_query, PandasReader
from burdock.metadata import CollectionMetadata
from burdock.metadata.collection import Table, Float

def test_sklearn_query():
   sklearn_dataset = sklearn.datasets.load_iris()
   sklearn_df = pd.DataFrame(data=sklearn_dataset.data, columns=sklearn_dataset.feature_names)


   iris = Table("dbo", "iris", 150, [
      Float("sepal length (cm)", 4, 8),
      Float("sepal width (cm)", 2, 5),
      Float("petal length (cm)", 1, 7),
      Float("petal width (cm)", 0, 3)
   ])
   schema = CollectionMetadata([iris], "csv")

   reader = PandasReader(schema, sklearn_df)
   rowset = execute_private_query(schema, reader, 0.3, 'SELECT AVG("petal width (cm)") FROM dbo.iris')
   df = pd.DataFrame(rowset[1:], columns=rowset[0])
   assert df is not None
   assert len(df) == 1
