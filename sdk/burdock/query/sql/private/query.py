from burdock.query.sql import QueryParser, Rewriter
from burdock.mechanisms.laplace import Laplace
import burdock.query.sql.ast.expressions.sql as ast
from burdock.query.sql.reader.rowset import TypedRowset

import numpy as np

"""
    Takes a rewritten query, executes against the target backend, then
    adds noise before returning the recordset.
"""
class PrivateQuery:
    def __init__(self, reader, metadata, epsilon=4.0):
        self.reader = reader
        self.metadata = metadata
        self.rewriter = Rewriter(metadata)
        self.epsilon = epsilon
        self.tau = 1

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
        query = self.rewriter.query(query)
        subquery = query.source.relations[0].primary.query
        return (subquery, query)

    def get_privacy_cost(self, query_string):
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

        # Preprocess:
        # 0. Rewrite query and execute subquery
        # 0b. Serialize to target backend
        subquery, query = self.rewrite_ast(query)

        syms = subquery.all_symbols()
        types = [s[1].type() for s in syms]
        sens = [s[1].sensitivity() for s in syms]

        # execute the subquery against the backend and load in typed rowset
        srs = self.reader.execute_typed(subquery)
        return (subquery, query, syms, types, sens, srs)

    def _postprocess(self, subquery, query, syms, types, sens, srs, pct=0.95):
        # Postprocess:
        # 1. Add Noise to subquery results
        # 1b. Clamp counts to 0, set SUM = NULL if count = 0
        # 2. Filter tau thresh
        # 3. Evaluate outer expression, set AVG = NULL if count = 0
        # 4. Sort        

        # # if user has selected keycount for outer query, use that instead
        kcc = [kc for kc in subquery.keycount_symbols() if kc[0] != "keycount"]
        if len(kcc) > 0:
            srs["keycount"] = srs[kcc[0][0].lower()]

        # add noise to all columns that need noise
        for nsym in subquery.numeric_symbols():
            name, sym = nsym
            name = name.lower()
            sens = sym.sensitivity()
            # treat null as 0 before adding noise
            srs[name] = np.array([v if v is not None else 0.0 for v in srs[name]])
            mechanism = Laplace(self.epsilon, sens, self.tau)
            srs.bounds[name] = mechanism.bounds(pct)
            srs[name] = mechanism.release(srs[name])
            # BUGBUG: Things other than counts can have sensitivity of 1
            if sym.sensitivity() == 1:
                counts = srs[name]
                counts[counts < 0] = 0
                srs[name] = counts

        if subquery.agg is not None:
            srs = srs.filter("keycount", ">", self.tau ** 2)

        syms = query.all_symbols()
        types = [s[1].type() for s in syms]
        sens = [s[1].sensitivity() for s in syms]
        colnames = [s[0] for s in syms]
        newrs = TypedRowset([colnames], types, sens)

        srsc = srs.m_cols
        bindings = dict((name.lower(), srsc[name]) for name in srsc.keys())

        cols = []
        for c in query.select.namedExpressions:
            cols.append(c.expression.evaluate(bindings))
        for idx in range(len(cols)):
            newrs[newrs.idxcol[idx]] = cols[idx]

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
