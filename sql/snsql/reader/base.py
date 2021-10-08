import pandas as pd


class Reader:
    ENGINE = None

    @property
    def engine(self):
        return self.ENGINE.lower() if self.ENGINE else None

    def execute(self, query, *ignore, accuracy: bool = False):
        raise NotImplementedError("Execute must be implemented on the inherited class")

    def _to_df(self, rows):
        # always assumes the first row is column names
        rows = list(rows)
        header = rows[0]
        if len(header) == 2 and isinstance(header[1], (list, tuple)):
            accuracy = True
        else:
            accuracy = False
        if len(rows) < 1:
            return None
        elif len(rows) < 2:
            if not accuracy:
                return pd.DataFrame(columns=header)
            else:
                return (pd.DataFrame(columns=header[0]), [pd.DataFrame(columns=h) for h in header[1]])
        else:
            if not accuracy:
                return pd.DataFrame(rows[1:], columns=header)
            else:
                result = []
                accuracies = [[] for a in header[1]]
                for result_row, acc in rows:
                    result.append(result_row)
                    for acc_row, idx in zip(acc, range(len(acc))):
                        accuracies[idx].append(acc_row)

                return [pd.DataFrame(result[1:], columns=result[0]),
                        [pd.DataFrame(a[1:], columns=a[0]) for a in accuracies]]

    def execute_df(self, query, *ignore, accuracy: bool = False):
        if not isinstance(query, str):
            raise ValueError("Please pass a string to this function.")

        return self._to_df(self.execute(query, accuracy=accuracy))
