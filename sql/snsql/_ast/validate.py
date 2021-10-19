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
        self.keycol = self.query.key_column

        checks = [
            func
            for func in dir(QueryConstraints)
            if callable(getattr(QueryConstraints, func)) and func.startswith("check_")
        ]
        rets = [getattr(self, check)() for check in checks]

    def check_aggregate(self):
        q = self.query

        """
        Look at each column
        """
        for s in q._select_symbols:
            subqueries = s.expression.xpath('//Query')
            if len(subqueries) > 0:
                raise ValueError("Subqueries in column expressions not supported: {str(sym)}")
            aggfuncs = s.expression.xpath('//AggFunction')
            n_aggs = len(aggfuncs)
            if n_aggs > 0:
                # It's a column that uses an aggregate, such as SUM, COUNT, etc.
                for ac in aggfuncs:
                    if not isinstance(ac.expression, (TableColumn, AllColumns)):
                        # should allow literals here?  any scalar that doesn't include a 
                        raise ValueError("We don't support aggregation over expressions: " + str(ac))
                    if ac.expression.type() not in ['int', 'float', 'bool'] and not ac.name == 'COUNT':
                        raise ValueError(f"Aggregations must be over numeric or boolean, got {ac.expression.type()} in {str(ac)}")
            else:
                # It's not an aggregate.  Must be grouping or literal
                tcs = s.expression.xpath('//TableColumn')
                n_tcs = len(tcs)
                if n_tcs > 0:
                    # expression must appear in grouping expressions
                    match = False
                    for ge in q._grouping_symbols:
                        if s.expression == ge.expression:
                            match = True
                    if not match:
                        raise ValueError(f"Column {s.name if s.name else ''} does not include an aggregate and is not included in GROUP BY: {str(s.expression)}")
                else:
                    # it's a bare/literal expression
                    pass


    def check_groupkey(self):
        agg = self.query.agg
        gc = agg.groupedColumns() if agg is not None else []
        keycol = self.keycol.colname.lower() if self.keycol is not None else None
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
            tcs = [s.expression for s in syms if type(s.expression) is TableColumn]
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
