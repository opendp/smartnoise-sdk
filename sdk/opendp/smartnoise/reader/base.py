import pandas as pd


class Reader:
    ENGINE = None

    @property
    def engine(self):
        return self.ENGINE

    def execute(self, query):
        raise NotImplementedError("Execute must be implemented on the inherited class")

    def _to_df(self, rows):
        # always assumes the first row is column names
        if len(rows) < 1:
            return None
        elif len(rows) < 2:
            return pd.DataFrame(columns=rows[0])
        else:
            return pd.DataFrame(rows[1:], columns=rows[0])

    def execute_df(self, query):
        if not isinstance(query, str):
            raise ValueError("Please pass a string to this function.")

        return self._to_df(self.execute(query))
