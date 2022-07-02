import os
import json

from .base import SqlReader, NameCompare, Serializer
from .engine import Engine


class BigQueryReader(SqlReader):
    """
        A dumb pipe that gets a rowset back from a database using
        a SQL string, and converts types to some useful subset
    """

    ENGINE = Engine.BIGQUERY

    def __init__(self, credentials_path=None, conn=None, **kwargs):
        super().__init__(self.ENGINE)

        self.conn = None
        if conn is not None:
            self.conn = conn
        else:
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                pass
            elif credentials_path is not None:
                self.credentials_path = credentials_path
            else:
                raise ValueError(
                    "No GCP service account credentials found. Provide the path to the JSON file in either:\n"
                    "a. credentials_path argument or\n"
                    "b. GOOGLE_APPLICATION_CREDENTIALS environment variable"
                )
            
            try:
                from google.cloud import bigquery
                from google.oauth2 import service_account
                self.api = google
            except:
                pass

    def execute(self, query, *ignore, accuracy:bool=False):
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        cnxn = self.conn
        if cnxn is None:
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                creds = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], strict=False)
                credentials = self.api.oauth2.service_account.Credentials.from_service_account_info(
                    creds
                )
            else:
                credentials = self.api.oauth2.service_account.Credentials.from_service_account_file(
                    self.credentials_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            cnxn = self.api.cloud.bigquery.Client(credentials=credentials, project=credentials.project_id)

        # use AST for this; just using string replace for now
        query = query.replace("RANDOM", "rand")
        result = cnxn.query(str(query)).result()
        if result.total_rows == 0:
            return []
        else:
            df = result.to_dataframe()
            col_names = [tuple(df.columns)]
            rows = [tuple(row) for row in df.values]
            return col_names + rows

class BigQueryNameCompare(NameCompare):
    def __init__(self, search_path=None):
        self.search_path = search_path if search_path is not None else ["public"]

    def schema_match(self, query, meta):
        if query.strip() == "" and meta in self.search_path:
            return True
        return self.identifier_match(query, meta)

    def identifier_match(self, query, meta):
        query = self.clean_escape(query)
        meta = self.clean_escape(meta)
        if query == meta:
            return True
        if self.is_escaped(meta) and meta.lower() == meta:
            meta = meta.lower().replace('"', "")
        #if self.is_escaped(query) and query.lower() == query:
        if query.lower() == query:
            query = query.lower().replace('"', "") # TODO: Replace Single quote replacement to backtick `
        return meta == query

    def strip_escapes(self, value):
        return value.replace('"', "").replace("`", "").replace("[", "").replace("]", "")

    def should_escape(self, identifier):
        if self.is_escaped(identifier):
            return False
        if identifier.lower() in self.reserved():
            return True
        if identifier.replace(" ", "") == identifier.lower():
            return False
        else:
            return True

class BigQuerySerializer(Serializer):
    def __init__(self):
        super().__init__()

