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

        self.mechanism = Laplace(self.epsilon, self.tau)

    def rewrite(self, query_string):
        queries = QueryParser(self.metadata).queries(query_string)
        if len(queries) > 1:
            raise ValueError("Too many queries provided.  We can only execute one query at a time.")
        elif len(queries) == 0:
            return []

        query = self.rewriter.query(queries[0])
        subquery = query.source.relations[0].primary.query
        return (subquery, query)

    def get_privacy_cost(self, query_string):
        subquery, query = self.rewrite(query_string)
        return subquery.numeric_symbols()

    def execute(self, query_string):
        exact_values = self._execute_exact(query_string)
        return self._apply_noise(*exact_values)

    def _apply_noise(self, subquery, query, syms, types, sens, srs):
        # if user has selected keycount for outer query, use that instead
        kcc = [kc for kc in subquery.keycount_symbols() if kc[0] != "keycount"]
        if len(kcc) > 0:
            srs["keycount"] = srs[kcc[0][0].lower()]
        srs = srs.filter("keycount", ">", self.tau ** 2)

        # add noise to all columns that need noise
        for nsym in subquery.numeric_symbols():
            name, sym = nsym
            name = name.lower()
            sens = sym.sensitivity()
            if sym.type() == "int":
                if sym.sensitivity() == 1:
                    counts = self.mechanism.count(srs[name])
                    counts[counts < 0] = 0
                    srs[name] = counts
                    srs = srs.filter(name, ">", self.tau)
                elif sens is not None:
                    srs[name] = self.mechanism.sum_int(srs[name], sens)
            elif sym.type() == "float" and sens is not None:
                srs[name] = self.mechanism.sum_float(srs[name], sens)
                
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

        return newrs.rows()
        

    def _execute_exact(self, query_string):
        if not isinstance(query_string, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")

        subquery, query = self.rewrite(query_string)

        # 0. Rewrite query and execute subquery
        # 0b. Serialize to target backend
        # 1. Add Noise to subquery
        # 1b. Clamp counts to 0, set SUM = NULL if count = 0
        # 2. Filter tau thresh
        # 3. Evaluate outer expression, set AVG = NULL if count = 0
        # 4. Sort

        syms = subquery.all_symbols()
        types = [s[1].type() for s in syms]
        sens = [s[1].sensitivity() for s in syms]

        # execute the subquery against the backend and load in typed rowset
        srs = self.reader.execute_typed(subquery)
        return (subquery, query, syms, types, sens, srs)

    def execute_typed(self, query):
        if isinstance(query, str):
            raise ValueError("Please pass ASTs to execute_typed.  To execute strings, use execute.")
