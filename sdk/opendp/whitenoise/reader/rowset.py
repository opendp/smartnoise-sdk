import numpy as np
import datetime
import operator

from opendp.whitenoise.report import Report


class TypedRowset:
    """
        Represents a typed rowset supporting the types used in differentially private
        query.
    """

    def __init__(self, rows, types):
        """
            Initialize rowset

            :param rows: A list of tuples representing rows, with the first tuple being the
                column names, and the rest being the rows of the rowset.
            :param types: A list of types for the columns
        """
        header = [c.lower() for c in rows[0]]
        body = rows[1:]
        self.n_rows = len(body)

        # Columns may be anonymous, in which case we generate a unique name
        prefix = "Col"
        cur_col = 1

        self.types = {}
        self.colnames = []
        self.report = Report()

        for idx in range(len(header)):
            cname = header[idx]
            while cname == "???" or cname in self.types:
                cname = prefix + str(cur_col)
                cur_col += 1
            self.types[cname] = types[idx]
            self.colnames.append(cname)
        # for quick lookup
        self.colidx = dict(list(zip(self.colnames, range(len(self.colnames)))))
        self.idxcol = dict(list(zip(range(len(self.colnames)), self.colnames)))

        self.m_cols = {}
        # Now read each row into separate columns
        tempcols = {}
        for cn in self.colnames:
            tempcols[cn] = []
        for row in body:
            for idx in range(len(row)):
                cn = self.idxcol[idx]
                tempcols[cn].append(row[idx])
        for cn in self.colnames:
            self[cn] = tempcols[cn]

        # verify all lengths the same
        if not all([len(self.m_cols[colname]) == len(body) for colname in self.colnames]):
            raise ValueError("Some columns have different number of rows!")

    def __str__(self):
        if len(self) == 0:
            return "(empty)"
        widths = [self.get_width(colname) + 3 for colname in self.colnames]
        header = [self.format_width(colname, width) for colname, width in zip(self.colnames, widths)]
        divider = ["-" * width for width in widths]
        rows = [header, divider]
        for idx in range(self.n_rows):
            row = [" " + self.format_width(self.m_cols[colname][idx], width - 1) for colname, width in zip(self.colnames, widths)]
            rows.append(row)
        return "\n".join([" " + "|".join(r) for r in rows])

    def __len__(self):
        return self.n_rows

    def __getitem__(self, key):
        if isinstance(key, str):
            key = self.unescape(key)
            return self.m_cols[key]
        elif type(key) is int:
            return self[self.idxcol[key]]
        else:
            raise ValueError("Must use string or int to index the rowset")

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = self.unescape(key)
            if self.n_rows == 0:
                self.n_rows = len(value)
            if len(value) != self.n_rows:
                raise ValueError("Trying to add column with {0} rows to rowset with {1} rows".format(len(value), self.n_rows))
            cn = key
            t = self.types[cn]
            if t == "string":
                self.m_cols[cn] = np.array([str(v) if v is not None else v for v in value])
            elif t == "boolean":
                self.m_cols[cn] = np.array([bool(v) if v is not None else v for v in value])
            elif t == "int":
                self.m_cols[cn] = np.array([int(v) if v is not None else v for v in value])
            elif t == "float":
                self.m_cols[cn] = np.array([float(v) if v is not None else v for v in value])
            elif t == "datetime":
                self.m_cols[cn] = np.array([datetime.datetime(v) if v is not None else None for v in value])
            else:
                raise ValueError("Trying to load unknown type " + t)
        elif type(key) is int:
            self[self.idxcol[key]] = value
        else:
            raise ValueError("Must index by string or int")

    def filter(self, colname, relation, value):
        ops = {'>': operator.gt,
               '<': operator.lt,
               '>=': operator.ge,
               '<=': operator.le,
               '=': operator.eq}
        col = self[colname]
        types = [self.types[name] for name in self.colnames]
        rows = [tuple(self.colnames)]
        for idx in range(self.n_rows):
            if ops[relation](col[idx], value):
                rows.append(tuple(self[name][idx] for name in self.colnames))
        filtered_rs = TypedRowset(rows, types)
        filtered_rs.report = self.report
        return filtered_rs

    def rows(self, header=True):
        return ([self.colnames] if header else []) + [tuple(self.m_cols[name][idx] for name in self.colnames) for idx in range(self.n_rows)]

    def get_width(self, colname):
        t = self.types[colname]
        if t == "string":
            return max([len(s) for s in self.m_cols[colname]])
        elif t == "boolean":
            return 6
        elif t == "datetime":
            return 10
        elif t == "int":
            return int(max([4] + [np.floor(np.log10(abs(n))) + 1 for n in self.m_cols[colname] if n != 0]))
        elif t == "float":
            return int(max([4] + [np.floor(np.log10(abs(n))) + 5 for n in self.m_cols[colname] if n != 0] ))
        else:
            raise ValueError("Unknown type: " + t)

    def format_width(self, val, width):
        if type(val) is float:
            str_val = "{0:.3f}".format(val)
        else:
            str_val = str(val)
        if len(str_val) > width:
            str_val = str_val[:width]
            padding = ""
        else:
            padding = " " * (width - len(str_val))
        if type(val) is float or type(val) is int:
            return padding[:(len(padding)-1)] + str_val + " "
        else:
            return str_val + padding

    def compare(self, other, ratio=0.05):
        for idx in range(len(self.colnames)):
            sval = self[idx]
            oval = other[idx]

            if self.types[self.idxcol[idx]] != other.types[other.idxcol[idx]]:
                return False
            if self.types[self.idxcol[idx]] in ["int", "float"]:
                lbound = np.multiply(sval, (1.0 - ratio))
                ubound = np.multiply(sval, (1.0 + ratio))
                if not all([l < o and u > o for l, u, o in zip(lbound, ubound, oval)]):
                    return False
            else:
                if not all([s == o for s, o in zip(sval, oval)]):
                    return False
        return True

    def unescape(self, value):
        # remove identifier escaping
        return value.replace('"', '').replace('`', '').replace('[', '').replace(']', '')
