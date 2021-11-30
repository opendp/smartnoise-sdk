from .tokens import *
from .expression import *
import copy
import warnings

"""
    AST for parsed Python Query Batch.  Allows validation, normalization,
    rewriting, and serialization.  Grammar is a strict subset of SQL-92.
    Lexer and parser token names borrowed from SparkSQL Grammar.
"""


class Batch(Sql):
    """A batch of queries"""

    def __init__(self, queries: List["Query"]) -> None:
        self.queries = queries

    def children(self):
        return self.queries


class Query(SqlRel):
    """A single query"""

    def __init__(self, select, source, where, agg, having, order, limit, metadata=None, privacy=None) -> None:
        self.select = select
        self.source = source
        self.where = where
        self.agg = agg
        self.having = having
        self.order = order
        self.limit = limit

        self.max_ids = None
        self.sample_max_ids = None
        self.row_privacy = None

        self._named_symbols = None
        self._select_symbols = None

        if metadata:
            self.load_symbols(metadata, privacy=privacy)

    def load_symbols(self, metadata, privacy=None):
        self.privacy = privacy
        # recursively load symbols for all relations
        relations = self.source.relations
        for r in relations:
            r.load_symbols(metadata, privacy=privacy)
        if not all([r.has_symbols() for r in relations]):
            return  # unable to load symbols

        tables = []
        for t in self.find_nodes(Table):
            # grab the first column in the table, to extract table metadata
            tables.append(t._select_symbols[0].expression)
        if len(tables) > 0:
            self.max_ids = max(tc.max_ids for tc in tables)
            self.sample_max_ids = any(tc.sample_max_ids for tc in tables)
            self.row_privacy = any(tc.row_privacy for tc in tables)
            self.censor_dims = any(tc.censor_dims for tc in tables)

        # get grouping expression symbols
        self._grouping_symbols = []
        if self.agg:
            self._grouping_symbols = []
            for ge in self.agg.groupingExpressions:
                try:
                    symb = ge.expression.symbol(relations)
                except ValueError as err: # Check if the expression has been aliased in the SELECT clause
                    if isinstance(ge.expression, Column):
                        expr = [
                            ne.expression for ne in self.select.namedExpressions
                            if ne.name and metadata.compare.identifier_match(ge.expression.name, ne.name)
                        ]
                        if len(expr) == 1:
                            symb = expr[0].symbol(relations)
                        else:
                            raise err
                    else:
                        raise err
                self._grouping_symbols.append(Symbol(symb))


        # get namedExpression symbols
        _symbols = []
        for ne in self.select.namedExpressions:
            if type(ne.expression) is not AllColumns:
                name = ne.column_name()

                _symbol_expr = ne.expression.symbol(relations)
                _symbol = Symbol(_symbol_expr, name)

                # annotate selects that reference GROUP BY column
                _symbol.is_grouping_column = False
                for ge in self._grouping_symbols:
                    if _symbol_expr == ge.expression:
                        _symbol.is_grouping_column = True

                # annotate key_counts
                _symbol.is_key_count = False
                if _symbol.is_count:
                    col = _symbol.expression.xpath_first("//AggFunction[@name='COUNT']")
                    if col:
                        if self.row_privacy:
                            _symbol.is_key_count = isinstance(col.expression, AllColumns)
                        else:
                            _symbol.is_key_count = col.is_key_count

                if self.privacy:
                    # add mechanism
                    _symbol.mechanism = None
                    mechanisms = self.privacy.mechanisms
                    epsilon = self.privacy.epsilon
                    delta = self.privacy.delta
                    if not _symbol.is_grouping_column:
                        sensitivity = _symbol.expression.sensitivity()
                        t = _symbol.expression.type()
                        if t in ['int', 'float'] and sensitivity is not None:
                            stat = 'count' if _symbol.is_count else 'sum'
                            if _symbol.is_key_count and self.censor_dims:
                                stat = 'threshold'
                            mech_class = mechanisms.get_mechanism(sensitivity, stat, t)
                            mech = mech_class(epsilon, delta=delta, sensitivity=sensitivity, max_contrib=self.max_ids)
                            if _symbol.is_key_count:
                                mech.delta = delta
                            _symbol.mechanism = mech

                _symbols.append(_symbol)

                # attach to named expression for xpath in accuracy.py
                ne.m_symbol = _symbol
            else:
                # It's SELECT *, expand out to all columns
                syms = ne.expression.all_symbols(relations)
                for sym in syms:
                    _symbols.append(Symbol(sym.expression, sym.name))

        self._select_symbols = _symbols
        self._named_symbols = {}

        for sym in self._select_symbols:
            if sym.name == "???":
                continue
            if sym.name in self._named_symbols:
                raise ValueError("SELECT has duplicate column names: " + name)
            self._named_symbols[sym.name] = sym

    def symbol(self, expression):
        """
            returns the expression for an output column in the SELECT statement.
            Query objects do not have aliases, so caller must strip alias first.
        """
        if not self.has_symbols():
            raise ValueError("Attempted to get symbol from query with no symbols loaded.")
        if type(expression) is not Column:
            raise ValueError(
                "Can only request output columns from a query: " + str(type(expression))
            )
        return self[expression.name]

    @property
    def m_symbols(self):
        warnings.warn("m_symbols has been renamed to _select_symbols")
        return self._select_symbols

    def numeric_symbols(self):
        return [s for s in self._select_symbols if s.expression.type() in ["int", "float"]]

    @property
    def key_column(self):
        # return TableColumn used as the primary key
        rsyms = self.source.relations[0].all_symbols(AllColumns())
        tcsyms = [r.expression for r in rsyms if type(r.expression) is TableColumn]
        keys = [tc for tc in tcsyms if tc.is_key]
        if len(keys) > 1:
            raise ValueError("We only know how to handle tables with one key: " + str(keys))
        if self.row_privacy:
            if len(keys) > 0:
                raise ValueError("Row privacy is set, but metadata specifies a private_id")
            else:
                return None
        elif self.row_privacy == False:
            if len(keys) < 1:
                raise ValueError("No private_id column specified, and row_privacy is False")
            else:
                return keys[0]
        else:
            # symbols haven't been loaded yet
            if len(keys) < 1:
                return None
            else:
                # kp = keys[0].split(".")
                # return kp[len(kp) - 1]
                return keys[0]

    def children(self) -> List[Any]:
        return [self.select, self.source, self.where, self.agg, self.having, self.order, self.limit]

    def evaluate(self, bindings):
        return [(ne.name, ne.expression.evaluate(bindings)) for ne in self.select.namedExpressions]


class Select(Sql):
    """Result Columns"""

    def __init__(self, quantifier, namedExpressions):
        self.quantifier = quantifier
        self.namedExpressions = Seq(namedExpressions)

    def functions(self):
        return [c for c in self.namedExpressions if type(c.expression) is AggFunction]

    def aggregates(self):
        return [f for f in self.functions() if f.is_aggregate()]

    def children(self):
        return [Token("SELECT"), self.quantifier, self.namedExpressions]


class From(Sql):
    """From"""

    def __init__(self, relations):
        self.relations = Seq(relations)

    def children(self):
        return [Token("FROM"), self.relations]


class Where(Sql):
    """Predicates."""

    def __init__(self, condition):
        self.condition = condition

    def children(self):
        return [Token("WHERE"), self.condition]


class Aggregate(Sql):
    """Group By"""

    def __init__(self, groupingExpressions):
        self.groupingExpressions = Seq(groupingExpressions)

    def groupedColumns(self):
        return [ge.expression for ge in self.groupingExpressions if type(ge.expression) == Column]

    def children(self):
        return [Token("GROUP"), Token("BY"), self.groupingExpressions]


class Having(Sql):
    """Having clause"""

    def __init__(self, condition):
        self.condition = condition

    def children(self):
        return [Token("HAVING"), self.condition]


class Order(Sql):
    """Order By"""

    def __init__(self, sortItems):
        self.sortItems = Seq(sortItems)

    def children(self):
        return [Token("ORDER"), Token("BY"), self.sortItems]

    def symbol(self, relations):
        return Order(self.sortItems.symbol(relations))


class Limit(Sql):
    """Limit"""

    def __init__(self, n):
        self.n = n

    def children(self):
        return [Token("LIMIT"), Literal(self.n, str(self.n))]

    def symbol(self, relations):
        return self


class Top(Sql):
    """Top"""

    def __init__(self, n):
        self.n = n

    def children(self):
        return [Token("TOP"), Literal(self.n, str(self.n))]

    def symbol(self, relations):
        return self


"""
    RELATIONS
"""


class Relation(SqlRel):
    """A relation such as table, join, or subquery"""

    def __init__(self, primary, joins):
        self.primary = primary
        self.joins = joins if joins is not None else []

    def load_symbols(self, metadata, privacy=None):
        self.privacy = privacy
        relations = [self.primary] + [j for j in self.joins]
        for r in relations:
            r.load_symbols(metadata, privacy)
        # check the join keys
        if len(self.joins) > 0:
            primary_symbols = [s.name.lower() for s in self.primary.all_symbols(AllColumns())]
            for j in self.joins:
                join_symbols = [s.name.lower() for s in j.right.all_symbols(AllColumns())]
                if type(j.criteria) is UsingJoinCriteria:
                    for i in j.criteria.identifiers:
                        if not i.name.lower() in primary_symbols:
                            raise ValueError(
                                "Join clause uses a join column that doesn't exist in the primary relation: "
                                + str(i)
                            )
                        if not i.name.lower() in join_symbols:
                            raise ValueError(
                                "Join clause uses a join column that doesn't exist in the joined relation: "
                                + str(i)
                            )

    def symbol(self, expression):
        if type(expression) is not Column:
            raise ValueError("Tables can only have column symbols: " + str(type(expression)))
        alias, colname = self.split_alias(expression.name)
        alias = alias if alias != "" else None
        syms_a = self.all_symbols(AllColumns(alias))
        syms_b = [s for s in syms_a if s is not None]
        syms_c = [
            s.expression
            for s in syms_b
            if (type(s.expression) is TableColumn and s.expression.compare.identifier_match(colname, s.name))
            or s.name == colname
        ]
        if len(syms_c) == 1:
            return syms_c[0]
        elif len(syms_c) > 1:
            raise ValueError("Too many relations matched column, ambiguous: " + str(expression))
        else:
            raise ValueError("Symbol could not be found in any relations: " + str(expression))

    def all_symbols(self, expression=None):
        if expression is None:
            expression = AllColumns()
        if type(expression) is not AllColumns:
            raise ValueError("Can only request all columns with * : " + str(type(expression)))
        syms = (
            self.primary.all_symbols(expression)
            if self.primary.alias_match(str(expression))
            else []
        )
        for j in self.joins:
            if not j.alias_match(str(expression)):
                continue
            drop_cols = []
            alias, name = self.split_alias(str(expression))
            # if alias.* specified, don't drop join column
            if type(j.criteria) is UsingJoinCriteria and alias == "":
                drop_cols = [str(i).lower() for i in j.criteria.identifiers]
            syms = syms + [
                Symbol(sym.expression, sym.name)
                for sym in j.all_symbols(expression)
                if sym.name.lower() not in drop_cols
            ]
        if len(syms) == 0:
            raise ValueError("Symbol could not be found in any relations: " + str(expression))
        return syms

    def children(self):
        return [self.primary] + self.joins


class Table(SqlRel):
    """A fully qualified table name with optional alias"""

    def __init__(self, name, alias):
        self.name = name
        self.alias = alias
        self._select_symbols = None
        self._named_symbols = None

    def symbol(self, expression):
        if type(expression) is not Column:
            raise ValueError("Tables can only have column symbols: " + str(type(expression)))
        if not self.alias_match(expression.name):
            raise ValueError(
                "Attempt to look up symbol with different alias.  Use alias_match() first."
                + expression.name
                + " -- "
                + str(self.name)
            )
        alias, name = self.split_alias(expression.name)
        if self._select_symbols is None:
            raise ValueError("Please load symbols with metadata first: " + str(self))
        else:
            if name in self:
                return self[name]
            else:
                return None

    def load_symbols(self, metadata, privacy=None):
        self.privacy = privacy
        self._named_symbols = None
        if metadata is None:
            return
        else:
            table = metadata[str(self.name)]
            if table is None:
                raise ValueError("No metadata available for " + str(self.name))
            tc = table.m_columns
            def get_table_expr(name):
                return TableColumn(
                    tablename=self.name,
                    colname=name,
                    valtype=tc[name].typename(),
                    is_key=tc[name].is_key,
                    lower=tc[name].lower if hasattr(tc[name], "lower") else None,
                    upper=tc[name].upper if hasattr(tc[name], "upper") else None,
                    nullable=tc[name].nullable if hasattr(tc[name], "nullable") else True,
                    missing_value=tc[name].missing_value if hasattr(tc[name], "missing_value") else None,
                    sensitivity=tc[name].sensitivity if hasattr(tc[name], "sensitivity") else None,
                    max_ids=table.max_ids,
                    sample_max_ids=table.sample_max_ids,
                    row_privacy=table.row_privacy,
                    censor_dims=table.censor_dims,
                    compare=metadata.compare
                )
            self._select_symbols = [Symbol(get_table_expr(name), name) for name in tc.keys()]

    def escaped(self):
        # is any part of this identifier escaped?
        parts = str(self).split(".")
        return any([p.startswith('"') or p.startswith("[") for p in parts])

    def children(self):
        return [self.name] + ([Token("AS"), self.alias] if self.alias is not None else [])


class AliasedSubquery(SqlRel):
    """A subquery with optional alias"""

    def __init__(self, query, alias):
        self.query = query
        self.alias = alias

    def symbol(self, expression):
        alias, name = self.split_alias(expression.name)
        return self.query.symbol(Column(name))

    def all_symbols(self, expression):
        if type(expression) is not AllColumns:
            raise ValueError("Need to pass in a * or alias.* to get all columns")
        if not self.alias_match(str(expression)):
            raise ValueError("Requesting all coluns with mismatched alias")
        return self.query.all_symbols(AllColumns())

    def children(self):
        return [Token("("), self.query, Token(")")] + (
            [Token("AS"), self.alias] if self.alias is not None else []
        )


class AliasedRelation(SqlRel):
    """A subrelation (table, join, or subquery) with optional alias"""

    def __init__(self, relation, alias):
        self.relation = relation
        self.alias = alias

    def symbol(self, expression):
        alias, name = self.split_alias(expression.name)
        return self.relation.symbol(Column(name))

    def all_symbols(self, expression):
        if type(expression) is not AllColumns:
            raise ValueError("Need to pass in a * or alias.* to get all columns")
        if not self.alias_match(str(expression)):
            raise ValueError("Requesting all coluns with mismatched alias")
        return self.relation.all_symbols(AllColumns())

    def children(self):
        return [Token("("), self.relation, Token(")")] + (
            [Token("AS"), self.alias] if self.alias is not None else []
        )


class Join(SqlRel):
    """A join expression attached to a primary relation"""

    def __init__(self, joinType, right, criteria):
        self.joinType = joinType
        self.right = right
        self.criteria = criteria

    def symbol(self, expression):
        return self.right.symbol(expression)

    def all_symbols(self, expression):
        return self.right.all_symbols(expression)

    def children(self):
        return [self.joinType, Token("JOIN"), self.right, self.criteria]


#
#    METADATA
#
class TableColumn(SqlExpr):
    """ A column attached to a fully qualified table """

    def __init__(
        self,
        tablename,
        colname,
        valtype="unknown",
        *ignore,
        is_key=False,
        lower=None,
        upper=None,
        max_ids=1,
        sample_max_ids=True,
        row_privacy=False,
        censor_dims=False,
        compare=None,
        nullable = True,
        missing_value = None,
        sensitivity = None
    ):
        self.tablename = tablename
        self.colname = colname
        self.valtype = valtype
        self.is_key = is_key
        self.lower = lower
        self.upper = upper
        self.max_ids = max_ids
        self.sample_max_ids = sample_max_ids
        self.row_privacy = row_privacy
        self.censor_dims = censor_dims
        self.unbounded = lower is None or upper is None
        self.nullable = nullable
        self.missing_value = missing_value
        self._sensitivity = sensitivity
        self.compare = compare

    def __str__(self):
        return self.tablename + "." + self.colname

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.tablename == other.tablename and self.colname == other.colname

    def __hash__(self):
        return hash((self.tablename, self.colname))

    def type(self):
        return self.valtype

    def sensitivity(self):
        if self.valtype in ["int", "float"]:
            if self.lower is not None and self.upper is not None:
                bounds_sensitivity = max(abs(self.upper), abs(self.lower))
                if self._sensitivity is not None:
                    return self._sensitivity
                else:
                    return bounds_sensitivity
            else:
                if self._sensitivity is not None:
                    return self._sensitivity
                else:
                    return np.inf  # unbounded
        elif self.valtype == "boolean":
            return 1
        else:
            return None

    def evaluate(self, bindings):
        if str(self).lower() in bindings:
            return bindings[str(self).lower()]
        else:
            return None

    @property
    def is_key_count(self):
        return self.is_key

    @property
    def is_count(self):
        return False
