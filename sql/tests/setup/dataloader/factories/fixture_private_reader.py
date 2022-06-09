from snsql.sql.parse import QueryParser
from snsql.sql.private_reader import PrivateReader
import copy


class FixturePrivateReader(PrivateReader):
    def __init__(self, reader, metadata, privacy=None):
        super().__init__(reader, metadata, privacy)
    def _execute_ast(self, query, *ignore, accuracy: bool = False, pre_aggregated=None, postprocess=True):
        return super()._execute_ast(query, accuracy=accuracy, pre_aggregated=pre_aggregated, postprocess=postprocess)
    @classmethod
    def from_connection(cls, conn, *ignore, privacy, metadata, engine=None, table_name=None, **kwargs):
        reader = super().from_connection(conn, *ignore, privacy=privacy, metadata=metadata, engine=engine, **kwargs)

        # make sure the table name is valid
        if table_name:
            parsed_table_name = QueryParser().parse_table_name(table_name)
            if parsed_table_name != table_name:
                raise ValueError(f"Attempting to create FixturePrivateReader with invalid table_name {table_name}")
        reader.table_name = table_name

        # update metadata to match table name
        new_metadata = copy.deepcopy(reader.metadata)

        table_parts = table_name.split('.')
        n_parts = len(table_parts)
        schema = None
        table = None
        if n_parts > 3:
            raise ValueError(f"Trying to update metadata with invalid table name {table_name}")
        elif n_parts == 3:
            new_metadata.dbname = table_parts[0]
            schema = table_parts[1]
            table = table_parts[2]
        elif n_parts == 2:
            schema = table_parts[0]
            table = table_parts[1]
        else: # 1 part
            table = table_parts[0]

        if len(new_metadata.m_tables) != 1:
            raise ValueError("FixturePrivateReader only works with single-table metadata")

        old_key = list(new_metadata.m_tables.keys())[0]
        new_table = copy.deepcopy(new_metadata[old_key])
        del new_metadata.m_tables[old_key]

        new_table.schema = schema
        new_table.name = table

        new_key = f"{schema}.{table}" if schema else table
        new_metadata.m_tables[new_key] = new_table

        reader.metadata = new_metadata
        reader._refresh_options()
        
        return reader
    def parse_query_string(self, query_string):
        # fix the query to match the table name
        query_ast = QueryParser().query(query_string)
        tables = query_ast.xpath('//Table')
        for table in tables:
            table.name = self.table_name
        return super().parse_query_string(str(query_ast))
