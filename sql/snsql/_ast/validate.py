from .ast import *
from snsql.metadata import Metadata


class Validate(object):
    """
        Checks a batch AST for any violations of our query requirements
        and returns error messages.
    """

    def validateBatch(self, batch, metadata):
        for q in batch:
            self.validateQuery(q, metadata)

    def validateQuery(self, query, metadata):
        """
            Checks the AST for a SELECT query to ensure conformance with
            requirements for differential privacy queries.
        """
        qc = QueryConstraints(query, metadata)
        qc.validate_all()


class QueryConstraints:
    """
        A collection of boolean functions that check for validity of
        a parsed Query AST.  Create the object by passing in the AST,
        then call any or all check functions.  Check functions pass
        or raise ValueError with descriptive message.
    """

    def __init__(self, query, metadata):
        self.query = query

        if metadata:
            self.metadata = Metadata.from_(metadata)
        else:
            self.metadata = metadata

    def validate_all(self):
        # will throw if more or less than one key
        self.keycol = self.key_col(self.query)

        checks = [
            func
            for func in dir(QueryConstraints)
            if callable(getattr(QueryConstraints, func)) and func.startswith("check_")
        ]
        rets = [getattr(self, check)() for check in checks]

    def check_aggregate(self):
        q = self.query

        def simplify(exp):
            while type(exp) is NestedExpression:
                exp = exp.expression
            if type(exp) is ArithmeticExpression:
                l = simplify(exp.left)
                r = simplify(exp.right)
                if type(l) is Literal:
                    exp = r
                elif type(r) is Literal:
                    exp = l
            if isinstance(exp, (PowerFunction, RoundFunction)):
                exp = exp.expression
            return exp

        select_expressions = [simplify(s[1]) for s in q.m_symbols]
        is_agg = lambda n: type(n) == AggFunction and n.is_aggregate
        aggs = [expr for expr in select_expressions if is_agg(expr)]
        non_aggs = [expr for expr in select_expressions if not is_agg(expr)]

        grouping_expressions = (
            [simplify(exp.expression) for exp in q.agg.groupingExpressions]
            if q.agg is not None
            else []
        )
        group_col_names = [gc.name for gc in q.agg.groupedColumns()] if q.agg is not None else []

        for nac in non_aggs:
            if not isinstance(nac, (TableColumn, Literal, BareFunction)):
                raise ValueError("Select column not a supported type: " + str(nac))
            if type(nac) is TableColumn:
                if nac.colname not in group_col_names:
                    raise ValueError(
                        "Attempting to select a column not in a GROUP BY clause: " + str(nac)
                    )

        for ac in aggs:
            if not isinstance(ac.expression, (TableColumn, AllColumns)):
                raise ValueError("We don't support aggregation over expressions: " + str(ac))
            if ac.expression.type() not in ['int', 'float', 'bool'] and not ac.name == 'COUNT':
                raise ValueError(f"Aggregations must be over numeric or boolean, got {ac.expression.type()} in {str(ac)}")

        for ge in grouping_expressions:
            if not isinstance(ge, Column):
                raise ValueError("We don't support grouping by expressions: " + str(ge))

    def check_groupkey(self):
        agg = self.query.agg
        gc = agg.groupedColumns() if agg is not None else []
        keycol = self.keycol.lower() if self.keycol is not None else None
        gbk = [g for g in gc if g.name.lower() == keycol]
        if (len(gbk) > 0) and (len(gbk) == len(gc)):
            raise ValueError(
                "GROUP BY must include more than key columns: " + ", ".join([str(g) for g in gc])
            )

    def check_select_relations(self):
        rel_nodes = self.query.select.find_nodes(SqlRel)
        if (len(rel_nodes)) > 0:
            raise ValueError("We don't support subqueries in the SELECT clause")

    def check_source_relations(self):
        relations = self.query.source.relations
        if len(relations) != 1:
            raise ValueError("Query must reference only one relation")

        relation = relations[0]
        self.walk_relations(relation)

    def walk_relations(self, r):
        if type(r) is AliasedSubquery:
            raise ValueError("Support for subqueries is currently disabled")
        if (
            type(r) is Query
            or type(r) is Table
            or type(r) is AliasedRelation
            or type(r) is AliasedSubquery
        ):
            syms = r.all_symbols(AllColumns())
            tcs = [s for name, s in syms if type(s) is TableColumn]
            if not any([tc.is_key for tc in tcs]):
                if not any([tc.row_privacy for tc in tcs]):
                    raise ValueError("Source relation must include a private key column: " + str(r))
        if type(r) is Join:
            raise ValueError("Support for JOIN queries is currently disabled")
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

        if query.row_privacy:
            if len(keys) > 0:
                raise ValueError("Row privacy is set, but metadata specifies a private_id")
            else:
                return None
        else:
            kp = keys[0].split(".")
            return kp[len(kp) - 1]
