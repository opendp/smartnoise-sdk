
from azureml.core import Dataset
from burdock.query.sql.reader.base import BaseReader
import burdock.query.sql.ast.ast as ast

"""
    A dumb pipe that gets a rowset back from a database using
    a SQL string, and converts types to some useful subset
"""
class DataReader(BaseReader):
    def __init__(self, datastore):
        self.datastore = datastore
        self.serializer = SqlServerSerializer()

    def update_connection_string(self):
        pass

    def switch_database(self, dbname):
        sql = "USE " + dbname + ""
        try:
            self.execute(sql)
        except:
            # Expected failure since the return is not a dataset
            pass

    def db_name(self):
        sql = "SELECT DB_NAME() as db_name"
        dbname = self.execute(sql)[1][0]
        return dbname

    def execute(self, sql_query):
        import pdb; pdb.set_trace()
        dataset = Dataset.Tabular.from_sql_query((self.datastore, sql_query))
        dataset_pd = dataset.to_pandas_dataframe()
        return [tuple([col for col in dataset_pd.columns])] + [val[1:] for val in dataset_pd.itertuples()]

class SqlServerSerializer:
    def serialize(self, query):
        for re in [n for n in query.find_nodes(ast.BareFunction) if n.name == 'RANDOM']:
            re.name = 'NEWID'

        for b in [n for n in query.find_nodes(ast.Literal) if isinstance(n.value, bool)]:
            b.text = "'TRUE'" if b.value else "'FALSE'"

        # T-SQL doesn't support USING critera, rewrite to ON
        for rel in [n for n in query.find_nodes(ast.Relation) if n.joins is not None and len(n.joins) > 0]:
            join_idx = 0
            for j in [j for j in rel.joins if isinstance(j.criteria, ast.UsingJoinCriteria)]:
                join_idx += 1
                ids = j.criteria.identifiers
                if rel.primary.alias is None:
                    rel.primary.alias = "PJXX"  # should use a name scope manager here
                if j.right.alias is None:
                    j.right.alias = "PJYY" + str(join_idx)  # should use naming scope

                left = [rel.primary.alias + "." + str(i) for i in ids]
                right = [j.right.alias + "." + str(i) for i in ids]
                frag = " AND ".join(l + " = " + r for l, r in zip(left, right))
                j.criteria = ast.BooleanJoinCriteria(ast.Expression(frag))

        return(str(query))
