import math
import numpy as np
from burdock.sql import Rewriter
from .parse import QueryParser
from burdock.mechanisms.laplace import Laplace
from burdock.mechanisms.gaussian import Gaussian
from burdock.metadata.report import Interval, Intervals, Result
from burdock.reader.sql.rowset import TypedRowset
from .ast.expressions import sql as ast

"""
    Takes a rewritten query, executes against the target backend, then
    adds noise before returning the recordset.
"""
class PrivateReader:
    def __init__(self, reader, metadata, epsilon=1.0, delta=10E-16, interval_widths=[0.95, 0.985], options=None):
        self.options = options if options is not None else PrivateReaderOptions()
        self.reader = reader
        self.metadata = metadata
        self.rewriter = Rewriter(metadata)
        self.epsilon = epsilon
        self.delta = delta
        self.max_contrib = 1
        self.interval_widths = interval_widths
        self.refresh_options()

    def refresh_options(self):
        self.rewriter = Rewriter(self.metadata)
        self.metadata.compare = self.reader.compare
        self.rewriter.options.reservoir_sample = self.options.reservoir_sample
        self.rewriter.options.clamp_columns = self.options.clamp_columns

    def parse_query_string(self, query_string):
        queries = QueryParser(self.metadata).queries(query_string)
        if len(queries) > 1:
            raise ValueError("Too many queries provided.  We can only execute one query at a time.")
        elif len(queries) == 0:
            return []
        return queries[0]

    def rewrite(self, query_string):
        query = self.parse_query_string(query_string)
        return self.rewrite_ast(query)

    def rewrite_ast(self, query):
        self.refresh_options()
        query = self.rewriter.query(query)
        subquery = query.source.relations[0].primary.query
        return (subquery, query)

    def get_privacy_cost(self, query_string):
        self.refresh_options()
        subquery, query = self.rewrite(query_string)
        return subquery.numeric_symbols()

    def execute(self, query_string):
        """Executes a query and returns a recordset that is differentially private.

        Follows ODBC and DB_API convention of consuming query as a string and returning
        recordset as tuples.  This is useful for cases where existing DB_API clients
        want to swap out API calls with minimal changes.

        :param query_string: A query string in SQL syntax
        :return: A recordset structured as an array of tuples, where each tuple
         represents a row, and each item in the tuple is typed.  The first row should
         contain column names.
        """
        trs = self.execute_typed(query_string)
        return trs.rows()

    def execute_typed(self, query_string):
        """Executes a query and returns a differentially private typed recordset.

        This is the typed version of execute.

        :param query_string: A query in SQL syntax
        :return: A typed recordset, where columns can be referenced by name.
        """
        query = self.parse_query_string(query_string)
        return self.execute_ast_typed(query)

    def _preprocess(self, query):
        if isinstance(query, str):
            raise ValueError("Please pass AST to _preprocess.")

        subquery, query = self.rewrite_ast(query)
        self.max_contrib = query.max_ids
        self.tau = self.max_contrib * (1- ( math.log(2 * self.delta / self.max_contrib) / self.epsilon  ))

        syms = subquery.all_symbols()
        source_col_names = [s[0] for s in syms]

        # list of sensitivities in column order
        sens = [s[1].sensitivity() for s in syms]

        # tell which ones are key counts, in column order
        is_key_count = [s[1].is_key_count for s in syms]

        # set sensitivity to None if the column is a grouping key
        if subquery.agg is not None:
            group_keys = [ge.expression.name if hasattr(ge.expression, 'name') else None for ge in subquery.agg.groupingExpressions]
        else:
            group_keys = []
        is_group_key = [colname in group_keys for colname in [s[0] for s in syms]]
        for idx in range(len(sens)):
            if is_group_key[idx]:
                sens[idx] = None

        kc_pos = None
        kcc_pos = []
        for idx in range(len(syms)):
            sname, sym = syms[idx]
            if sname == 'keycount':
                kc_pos = idx
            elif sym.is_key_count:
                kcc_pos.append(idx)
        if kc_pos is None and len(kcc_pos) > 0:
            kc_pos = kcc_pos.pop()

        # make a list of mechanisms in column order
        mechs = [Gaussian(self.epsilon, self.delta, s, self.max_contrib, self.interval_widths) if s is not None else None for s in sens]

        # execute the subquery against the backend and load in typed rowset
        db_rs = self.reader.execute_ast_typed(subquery).rows()

        def process_row(row):
            for idx in range(len(row)):
                if sens[idx] is not None and row[idx] is None:
                    row[idx] = 0.0
            out_row = [noise.release([v]).values[0] if noise is not None else v for noise, v in zip(mechs, row)]
            for idx in kcc_pos:
                out_row[idx] = out_row[kc_pos]
            if self.options.clamp_counts:
                for idx in range(len(row)):
                    if is_key_count[idx] and row[idx] < 0:
                        row[idx] = 0

            return out_row


        out = map(process_row, db_rs[1:])

        if subquery.agg is not None and self.options.censor_dims:
            out = filter(lambda row: row[kc_pos] > self.tau, out)


        # get column information for outer query
        out_syms = query.all_symbols()
        out_types = [s[1].type() for s in out_syms]
        out_sens = [s[1].sensitivity() for s in out_syms]
        out_colnames = [s[0] for s in out_syms]

        def process_out_row(row):
            #raise ValueError("foo")
            bindings = dict((name.lower(), val ) for name, val in zip(source_col_names, row))
            return [c.expression.evaluate(bindings) for c in query.select.namedExpressions]
    
        out_new = map(process_out_row, out)

        # make the new recordset
        newrs = TypedRowset([out_colnames] + list(out_new), out_types, out_sens)         

#        for idx in range(len(cols)):
#            colname = newrs.idxcol[idx]
#            newrs[colname] = cols[idx]
#            newrs.report[colname] = Result(None, None, None, cols[idx], None, None, None, None, None, Intervals(intervals_list[idx]), None)

            #newrs.intervals[colname] = Intervals(intervals_list[idx])

        # Now sort, if it has order by clause
        if query.order is not None:
            sort_fields = []
            for si in query.order.sortItems:
                if type(si.expression) is not ast.Column:
                    raise ValueError("We only know how to sort by column names right now")
                colname = si.expression.name.lower()
                desc = False
                if si.order is not None and si.order.lower() == "desc":
                    desc = True
                sf = (colname, desc)
                sort_fields.append(sf)
            sf = [("-" if desc else "") + colname for colname, desc in sort_fields]
            newrs.sort(sf)
        return newrs

    def execute_ast(self, query):
        """Executes an AST representing a SQL query

        :param query: A SQL query in AST form
        :return: A recordset formatted as tuples for rows, with first row having column names
        """
        if isinstance(query, str):
            raise ValueError("Please pass ASTs to execute_typed.  To execute strings, use execute.")
        trs = self.execute_ast_typed(query)
        return trs.rows()

    def execute_ast_typed(self, query):
        """Executes an AST representing a SQL query, returning typed recordset

        :param query: A SQL query in AST form
        :return: A typed recordset
        """
        if isinstance(query, str):
            raise ValueError("Please pass ASTs to execute_typed.  To execute strings, use execute.")
        #subquery_results = self._preprocess(query)
        #return self._postprocess(*subquery_results)

        return self._preprocess(query)

class PrivateReaderOptions:
    """Options that control privacy behavior"""
    def __init__(self, 
        censor_dims=True, 
        clamp_counts=True, 
        reservoir_sample=True,
        clamp_columns=True,
        row_privacy=False):
        """Initialize with options.
        :param censor_dims: boolean, set to False if you know that small dimensions cannot expose privacy
        :param clamp_counts: boolean, set to False to allow noisy counts to be negative
        :param reservoir_sample: boolean, set to False if the data collection will never have more than max_contrib record per individual
        :param clamp_columns: boolean, set to False to allow values that exceed lower and higher limit specified in metadata.  May impact privacy
        :param row_privacy: boolean, True if each row is a separate individual"""
        self.censor_dims = censor_dims
        self.clamp_counts = clamp_counts
        self.reservoir_sample = reservoir_sample
        self.clamp_columns = clamp_columns
        self.row_privacy = row_privacy
