import math
import numpy as np
from .private_rewriter import Rewriter
from .parse import QueryParser
from .ast.expressions import sql as ast

from burdock.mechanisms.laplace import Laplace
from burdock.mechanisms.gaussian import Gaussian
from burdock.metadata.report import Interval, Intervals, Result
from burdock.reader.rowset import TypedRowset

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
        types = [s[1].type() for s in syms]
        sens = [s[1].sensitivity() for s in syms]

        # execute the subquery against the backend and load in typed rowset
        db_rs = self.reader.execute_ast_typed(subquery)
        return (subquery, query, syms, types, sens, db_rs)

    def _postprocess(self, subquery, query, syms, types, sens, db_rs):
        # add noise to all columns that need noise
        for nsym in subquery.numeric_symbols():
            name, sym = nsym
            name = name.lower()
            sens = sym.sensitivity()

            #if sym.expression in group_expressions:
            #    continue

            # treat null as 0 before adding noise
            db_rs[name] = np.array([v if v is not None else 0.0 for v in db_rs[name]])

            mechanism = Gaussian(self.epsilon, self.delta, sens, self.max_contrib, self.interval_widths)
            report = mechanism.release(db_rs[name], compute_accuracy=True)

            db_rs[name] = report.values
            db_rs.report[name] = report

            if (self.options.clamp_counts is True) and sym.is_key_count:
                counts = db_rs[name]
                counts[counts < 0] = 0
                db_rs[name] = counts

        # make sure all keycounts report same noisy values
        kcc = [kc for kc in subquery.keycount_symbols() if kc[0] != "keycount"]
        for kc in kcc:
            db_rs[kc[0].lower()] = db_rs["keycount"] = db_rs[kcc[0][0].lower()]
            

        # censor dimensions for privacy
        if subquery.agg is not None and self.options.censor_dims:
            db_rs = db_rs.filter("keycount", ">", self.tau)

        # get column information for outer query
        syms = query.all_symbols()
        types = [s[1].type() for s in syms]
        sens = [s[1].sensitivity() for s in syms]
        colnames = [s[0] for s in syms]

        db_rsc = db_rs.m_cols

        bindings_list = []

        # first do the noisy values
        bindings_list.append(dict((name.lower(), db_rsc[name]) for name in db_rsc.keys()))

        # now evaluate all lower and upper
        interval_widths = None
        for name in db_rsc.keys():
            alpha_list = db_rs.report[name].interval_widths if name in db_rs.report else None
            if alpha_list is not None:
                interval_widths = alpha_list
                break
        if interval_widths is not None:
            for confidence in interval_widths:
                bind_low = {}
                bind_high = {}
                for name in db_rsc.keys():
                    if name in db_rs.report and db_rs.report[name].intervals is not None:
                        bind_low[name.lower()] = db_rs.report[name].intervals[confidence].low
                        bind_high[name.lower()] = db_rs.report[name].intervals[confidence].high
                    else:
                        bind_low[name.lower()] = db_rsc[name]
                        bind_high[name.lower()] = db_rsc[name]
                bindings_list.append(bind_low)
                bindings_list.append(bind_high)
    
        cols = []
        intervals_list = []
        for c in query.select.namedExpressions:
            cols.append(c.expression.evaluate(bindings_list[0]))

            ivals = []
            # initial hack; just evaluate lower and upper for each confidence
            if interval_widths is not None:
                for idx in range(len(interval_widths)):
                    low_idx = idx * 2 + 1
                    high_idx = idx * 2 + 2
                    low = c.expression.evaluate(bindings_list[low_idx])
                    high = c.expression.evaluate(bindings_list[high_idx])
                    ivals.append(Interval(interval_widths[idx], None, low, high))
            intervals_list.append(ivals)

        # make the new recordset
        newrs = TypedRowset([colnames], types, sens)            
        for idx in range(len(cols)):
            colname = newrs.idxcol[idx]
            newrs[colname] = cols[idx]
            newrs.report[colname] = Result(None, None, None, cols[idx], None, None, None, None, None, Intervals(intervals_list[idx]), None)

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
        subquery_results = self._preprocess(query)
        return self._postprocess(*subquery_results)

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
