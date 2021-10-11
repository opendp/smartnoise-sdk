import random
import string

from snsql.metadata import Metadata

from .parse import QueryParser

from snsql._ast.validate import Validate
from snsql._ast.ast import (
    Select,
    From,
    Query,
    Where,
    Order,
    Literal,
    Column,
    TableColumn,
    AllColumns,
    NamedExpression,
    NestedExpression,
    AggFunction,
    MathFunction,
    ArithmeticExpression,
    BooleanCompare,
    FuncName,
    Op,
    Identifier,
    Token,
    CaseExpression,
    RankingFunction,
    OverClause,
    BareFunction,
    AliasedSubquery,
    Relation,
    SortItem,
    WhenExpression,
    Sql,
)


class Rewriter:
    """
        Modifies parsed ASTs to augment with information needed
        to support differential privacy.  Uses a matching
        object which contains metadata necessary for differential
        privacy, such as min/max, cardinality, and key columns.

        This routine is intended to be used prior to post-processing
        with random noise generation.

    """

    def __init__(self, metadata):
        self.options = RewriterOptions()
        self.metadata = Metadata.from_(metadata)

    def calculate_avg(self, exp, scope):
        """
            Takes an expression for a noisy mean and rewrites
            to a noisy sum and a noisy count
        """
        expr = exp.expression
        quant = exp.quantifier

        sum_expr = self.push_sum_or_count(AggFunction(FuncName("SUM"), quant, expr), scope)
        count_expr = self.push_sum_or_count(AggFunction(FuncName("COUNT"), quant, expr), scope)

        new_exp = NestedExpression(ArithmeticExpression(sum_expr, Op("/"), count_expr))
        return new_exp

    def calculate_variance(self, exp, scope):
        """
            Calculate the variance from avg of squares and square of averages
        """
        expr = exp.expression
        quant = exp.quantifier

        sum_expr = self.push_sum_or_count(AggFunction(FuncName("SUM"), quant, ArithmeticExpression(expr, Op("*"), expr)), scope)
        count_expr = self.push_sum_or_count(AggFunction(FuncName("COUNT"), quant, expr), scope)

        avg_of_square = NestedExpression(ArithmeticExpression(sum_expr, Op("/"), count_expr))


        avg = self.calculate_avg(AggFunction(FuncName("AVG"), quant, expr), scope)
        avg_squared = ArithmeticExpression(avg, Op("*"), avg)

        new_exp = ArithmeticExpression(avg_of_square, Op("-"), avg_squared)
        return new_exp

    def calculate_stddev(self, exp, scope):
        """
            Calculate the standard deviation from the variance
        """
        expr = AggFunction(FuncName("STD"), exp.quantifier, exp.expression)
        var_expr = self.calculate_variance(expr, scope)

        new_exp = MathFunction(FuncName("SQRT"), var_expr)
        return new_exp

    def push_sum_or_count(self, exp, scope):
        """
            Push a sum or count expression to child scope
            and convert to a sum
        """
        new_name = scope.push_name(AggFunction(exp.name, exp.quantifier, exp.expression))

        new_exp = Column(new_name)
        return new_exp

    def rewrite_agg_expression(self, agg_exp, scope):
        """
            rewrite AVG, VAR, etc. and push all sum or count
            to child scope
        """
        child_agg_exps = agg_exp.find_nodes(AggFunction)
        if len(child_agg_exps) > 0:
            raise ValueError("Cannot have nested aggregate functions: " + str(agg_exp))

        agg_func = agg_exp.name
        if agg_func in ["SUM", "COUNT"]:
            new_exp = self.push_sum_or_count(agg_exp, scope)
        elif agg_func == "AVG":
            new_exp = self.calculate_avg(agg_exp, scope)
        elif agg_func in ["VAR", "VARIANCE"]:
            new_exp = self.calculate_variance(agg_exp, scope)
        elif agg_func in ["STD", "STDDEV"]:
            new_exp =  self.calculate_stddev(agg_exp, scope)
        else:
            raise ValueError("We don't know how to rewrite aggregate function: " + str(agg_exp))
        return NestedExpression(new_exp)

    def rewrite_outer_named_expression(self, ne, scope):
        """
            look for all the agg functions and rewrite them,
            preserving all other portions of expression
        """
        name = ne.name
        exp = ne.expression

        if type(exp) is Column:
            new_name = scope.push_name(Column(exp.name))
            exp.name = new_name

        elif type(exp) is AggFunction:
            exp = self.rewrite_agg_expression(exp, scope)

        else:
            for outer_col_exp in exp.find_nodes(Column, AggFunction):
                new_name = scope.push_name(Column(outer_col_exp.name))
                outer_col_exp.name = new_name

            def replace_agg_exprs(expr):
                for child_name, child_expr in expr.__dict__.items():
                    if isinstance(child_expr, Sql):
                        replace_agg_exprs(child_expr)
                    if isinstance(child_expr, AggFunction):
                        expr.__dict__[child_name] = self.rewrite_agg_expression(child_expr, scope)

            replace_agg_exprs(exp)

        return NamedExpression(name, exp)

    def query(self, query):
        query = QueryParser(self.metadata).query(str(query))
        Validate().validateQuery(query, self.metadata)

        child_scope = Scope()

        if self.options.row_privacy:
            keycount_expr = AggFunction(FuncName("COUNT"), None, AllColumns())
        else:
            key_col = self.key_col(query)
            keycount_expr = AggFunction(FuncName("COUNT"), Token("DISTINCT"), Column(key_col))

        if self.options.censor_dims:
            child_scope.push_name(keycount_expr, "keycount")

        # we make sure aggregates are in select scope for subqueries
        if query.agg is not None:
            for ge in query.agg.groupingExpressions:
                child_scope.push_name(ge.expression)

        select = [
                self.rewrite_outer_named_expression(ne, child_scope)
                for ne in query.select.namedExpressions
            ]

        select = Select(query.select.quantifier, select)

        subquery = Query(
            child_scope.select(), query.source, query.where, query.agg, None, None, None
        )
        subquery = self.exact_aggregates(subquery)
        subquery = [Relation(AliasedSubquery(subquery, Identifier("exact_aggregates")), None)]
        return Query(select, From(subquery), None, query.agg, query.having, query.order, query.limit, metadata=self.metadata)

    def exact_aggregates(self, query):
        child_scope = Scope()

        for ne in query.select.namedExpressions:
            child_scope.push_name(ne.expression)

        select = [ne for ne in query.select.namedExpressions]
        select = Select(None, select)

        subquery = Query(
            child_scope.select(), query.source, query.where, query.agg, None, None, None
        )
        if self.options.reservoir_sample and not self.options.row_privacy:
            subquery = self.per_key_random(subquery)
            subquery = [Relation(AliasedSubquery(subquery, Identifier("per_key_random")), None)]

            filtered = Where(
                BooleanCompare(
                    Column("per_key_random.row_num"),
                    Op("<="),
                    Literal(str(self.options.max_contrib), self.options.max_contrib),
                )
            )
            return Query(select, From(subquery), filtered, query.agg, None, None, None)
        else:
            subquery = self.per_key_clamped(subquery)
            subquery = [Relation(AliasedSubquery(subquery, Identifier("per_key_all")), None)]
            return Query(select, From(subquery), None, query.agg, None, None, None)

    def per_key_random(self, query):
        key_col = self.key_col(query)

        select = [
                NamedExpression(None, AllColumns()),
                NamedExpression(
                    Identifier("row_num"),
                    RankingFunction(FuncName("ROW_NUMBER"),
                                    OverClause(
                                        Column(key_col),
                                        Order([
                                            SortItem(BareFunction(FuncName("RANDOM")), None)
                                            ])
                                    ),
                    ),
                ),
            ]
        select = Select(None, select)

        subquery = self.per_key_clamped(query)
        subquery = [
            Relation(AliasedSubquery(subquery, Identifier("clamped" if self.options.clamp_columns else "not_clamped")), None)
        ]

        return Query(select, From(subquery), None, None, None, None, None)

    def per_key_clamped(self, query):
        child_scope = Scope()
        key_col = self.key_col(query)
        if self.options.reservoir_sample and not self.options.row_privacy:
            child_scope.push_name(Column(key_col), key_col)
        relations = query.source.relations
        select = [
                self.clamp_expression(ne, relations, child_scope, query, self.options.clamp_columns)
                for ne in query.select.namedExpressions
            ]
        select = Select(None, select)
        if not child_scope.expressions:
            # nothing is selected, may be lone COUNT(*)
            if len(select.namedExpressions) == 1:
                expr = select.namedExpressions[0].expression
                if isinstance(expr, AggFunction) and expr.name == 'COUNT' and isinstance(expr.expression, AllColumns):
                    table = expr.expression.table
                    alias = '*' if table is None else '*_' + str(table)
                    child_scope.push_name(AllColumns(), alias)
        subquery = Query(child_scope.select(), query.source, query.where, None, None, None, None)
        return subquery

    def clamp_expression(self, ne, relations, scope, query, do_clamp=True):
        """
            Lookup the expression referenced in each named expression and
            write a clamped select for it, using the schema
        """
        def bounds_clamp(colname):
            if not do_clamp:
                return None, None
            if query.agg:
                grouping_colnames = [col.name for col in query.agg.groupedColumns()]
                if colname in grouping_colnames:
                    return None, None
            minval = None
            maxval = None
            sym = col.symbol(relations)
            if sym.valtype in ["float", "int"] and not sym.unbounded:
                minval = sym.minval
                maxval = sym.maxval
            if minval is None or maxval is None or sym.is_key:
                return None, None
            return minval, maxval

        exp = ne.expression
        cols = exp.find_nodes(Column)
        if type(exp) is Column:
            cols += [exp]
        for col in cols:
            colname = col.name
            minval, maxval = bounds_clamp(colname)
            if minval == None:
                cexpr = Column(colname)
                ce_name = scope.push_name(cexpr, str(colname))
            else:
                when_min = WhenExpression(
                    BooleanCompare(col, Op("<"), Literal(minval)), Literal(minval)
                    )
                when_max = WhenExpression(
                    BooleanCompare(col, Op(">"), Literal(maxval)), Literal(maxval)
                    )
                cexpr = CaseExpression(None, [when_min, when_max], col)
                ce_name = scope.push_name(cexpr, str(colname))
            col.name = ce_name
        return ne

    def key_col(self, query):
        """
            Return the key column, given a from clause
        """
        rsyms = query.source.relations[0].all_symbols(AllColumns())
        tcsyms = [r for name, r in rsyms if type(r) is TableColumn]
        keys = [str(tc) for tc in tcsyms if tc.is_key]
        if len(keys) > 1:
            raise ValueError("We only know how to handle tables with one key: " + str(keys))
        if self.options.row_privacy:
            if len(keys) > 0:
                raise ValueError("Row privacy is set, but metadata specifies a private_id")
            else:
                return None
        else:
            if len(keys) < 1:
                raise ValueError("No private_id column specified, and row_privacy is not set")
            else:
                kp = keys[0].split(".")
                return kp[len(kp) - 1]


class Scope:
    """
        A name scope for a select query
    """

    def __init__(self):
        self.expressions = {}

    def select(self, quantifier=None):
        return Select(
            quantifier,
            [
                NamedExpression(
                    Identifier(str(name)) if not str(name).startswith('*') else None, 
                    self.expressions[name]
                ) 
                for name in self.expressions.keys()
            ],
        )

    def push_name(self, expression, proposed=None):
        """
            Returns a named expression from an expression, using
            an existing name if already provided in this scope,
            or generating a new name and adding to the names
            dictionary if the expression does not exist in scope.
        """
        # see if the proposed name is taken
        if proposed is not None:
            if proposed in self.expressions:
                if self.expressions[proposed] == expression:
                    return proposed
                else:
                    pass
            else:
                self.expressions[proposed] = expression
                return proposed

        # see if the expression has been used
        names = [name for name in self.expressions.keys() if self.expressions[name] == expression]
        if len(names) > 0:
            return names[0]

        # see if the expression has been used under the symbol name
        proposed = expression.symbol_name()
        if proposed in self.expressions:
            if self.expressions[proposed] == expression:
                return proposed
            else:
                pass
        else:
            self.expressions[proposed] = expression
            return proposed

        # Expression hasn't been used, but name is taken. Generate random.
        while not proposed in self.expressions:
            proposed = "".join(random.choice(string.ascii_letters) for i in range(7))

        self.expressions[proposed] = expression
        return proposed


class RewriterOptions:
    """Options that modify rewriter behavior"""

    def __init__(self, row_privacy=False, reservoir_sample=True, clamp_columns=True, max_contrib=1, censor_dims=True):
        """Initialize options before running the rewriter

        :param row_privacy: boolean, True if each row is a separate individual
        :param reservoir_sample: boolean, set to False if the data collection will never have more than max_contrib record per individual
        :param clamp_columns: boolean, set to False to allow values that exceed lower and higher limit specified in metadata.  May impact privacy
        :param max_contrib: int, set to maximum number of individuals that can appear in any query result.  Affects sensitivity
        :param censor_dims: boolean, tells whether or not to censor infrequent dimensions
        """
        self.row_privacy = row_privacy
        self.reservoir_sample = reservoir_sample
        self.clamp_columns = clamp_columns
        self.max_contrib = max_contrib
        self.censor_dims = censor_dims