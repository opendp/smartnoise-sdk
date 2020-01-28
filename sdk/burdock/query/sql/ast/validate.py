from .ast import *

class Validate(object):
    """
        Checks a batch AST for any violations of our query requirements
        and returns error messages.
    """
    def validateBatch(self, batch, metadata):
        for q in batch:
            self.validateQuery(q, metadata)

    """
        Checks the AST for a SELECT query to ensure conformance with
        requirements for differential privacy queries.
    """
    def validateQuery(self, query, metadata):
        qc = QueryConstraints(query, metadata)
        qc.checkAll()


"""
    A collection of boolean functions that check for validity of
    a parsed Query AST.  Create the object by passing in the AST,
    then call any or all check functions.  Each check function returns
    either (False, "Message") or (True, "")
"""
class QueryConstraints:
    def __init__(self, query, metadata):
        self.query = query
        self.metadata = metadata


    def checkAll(self):
        # will throw if more or less than one key
        self.keycol = self.key_col(self.query)

        checks = [func for func in dir(QueryConstraints) if callable(getattr(QueryConstraints, func)) and func.startswith("check_")]
        rets = [getattr(self, check)() for check in checks]

    def check_aggregate(self):
        nes = self.query.select.namedExpressions
        agg = self.query.agg
        gc = agg.groupedColumns() if agg is not None else []
        exp = [c.expression for c in nes]
        agg = [e for e in exp if type(e) == AggFunction and e.is_aggregate()]

#        no_agg = [e for e in exp if e not in agg and e not in gc]
#        if (len(no_agg)) > 0:
#            raise ValueError("Query cannot return non-aggregate columns: " + ", ".join([str(n) for n in no_agg]))
#        if (len(agg)) == 0:
#            raise ValueError("Query must return at least one aggregate value")


    def check_groupkey(self):
        agg = self.query.agg
        gc = agg.groupedColumns() if agg is not None else []
        gbk = [g for g in gc if g.name.lower() == self.keycol.lower()]
        if (len(gbk) > 0) and (len(gbk) == len(gc)):
            raise ValueError("GROUP BY must include more than key columns: " + ", ".join([str(g) for g in gc]))

    def check_source_relations(self):
        relations = self.query.source.relations
        if len(relations) != 1:
            raise ValueError("Query must reference only one relation")

        relation = relations[0]
        self.walk_relations(relation)

    def walk_relations(self, r):
        if type(r) is Query or type(r) is Table or type(r) is AliasedRelation or type(r) is AliasedSubquery:
            syms = r.all_symbols(AllColumns())
            tcs = [s for name, s in syms if type(s) is TableColumn ]
            if not any([tc.is_key for tc in tcs]):
                raise ValueError("Source relation must include a private key column: " + str(r))
        if type(r) is Join:
            if type(r.criteria) is not UsingJoinCriteria:
                raise ValueError("We only support JOIN with USING semantics currently")
            ids = [str(i).lower() for i in r.criteria.identifiers]
            if self.keycol.lower() not in ids:
                print(ids)
                print(self.keycol)
                raise ValueError("All JOINS must include the private key")
        for c in [ch for ch in r.children() if ch is not None]:
            self.walk_relations(c)

    """
        Return the key column, given a from clause
    """
    def key_col(self, query):
        rsyms = query.source.relations[0].all_symbols(AllColumns())
        tcsyms = [r for name, r in rsyms if type(r) is TableColumn]
        keys = [str(tc) for tc in tcsyms if tc.is_key]
        if len(keys) > 1:
            raise ValueError("We only know how to handle tables with one key: " + str(keys))
        elif len(keys) < 1:
            raise ValueError("No key column available in query relations")

        kp = keys[0].split(".")
        return kp[len(kp) - 1]
