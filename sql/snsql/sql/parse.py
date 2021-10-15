from snsql.metadata import Metadata
from .parser.SqlSmallLexer import SqlSmallLexer  # type: ignore
from .parser.SqlSmallParser import SqlSmallParser  # type: ignore
from .parser.SqlSmallVisitor import SqlSmallVisitor  # type: ignore
from .parser.SqlSmallErrorListener import SyntaxErrorListener  # type: ignore

from antlr4 import *  # type: ignore
from snsql._ast.tokens import *
from snsql._ast.ast import *


class QueryParser:
    def __init__(self, metadata=None):
        if metadata:
            self.metadata = Metadata.from_(metadata)
        else:
            self.metadata = None

    def start_parser(self, stream):
        lexer = SqlSmallLexer(stream)
        stream = CommonTokenStream(lexer)
        parser = SqlSmallParser(stream)
        parser._interp.predictionMode = PredictionMode.LL_EXACT_AMBIG_DETECTION
        lexer._listeners = [SyntaxErrorListener(), DiagnosticErrorListener()]
        parser._listeners = [SyntaxErrorListener(), DiagnosticErrorListener()]
        return parser

    def queries(self, query_string, metadata=None):
        if metadata is None and self.metadata is not None:
            metadata = self.metadata
        elif metadata:
            metadata = Metadata.from_(metadata)

        istream = InputStream(query_string)
        parser = self.start_parser(istream)
        bv = BatchVisitor()
        queries = [q for q in bv.visit(parser.batch()).queries]
        if metadata is not None:
            for q in queries:
                q.load_symbols(metadata)
        return queries

    def query(self, query_string, metadata=None):
        queries = self.queries(query_string, metadata)
        if len(queries) > 1:
            raise ValueError("Attempt to parse query resulted in a batch with more than one")
        q = queries[0]
        if metadata is not None:
            q.load_symbols(metadata)
        return q

    def parse_only(self, query_string):
        if query_string.strip().upper().startswith("SELECT") or query_string.strip().startswith(
            "--"
        ):
            istream = InputStream(query_string)
        else:
            istream = FileStream(query_string)
        parser = self.start_parser(istream)
        SqlSmallVisitor().visit(parser.batch())
        return None

    def parse_named_expressions(self, expression_string):
        istream = InputStream(expression_string)
        parser = self.start_parser(istream)
        nev = NamedExpressionVisitor()
        return nev.visitNamedExpressionSeq(parser.namedExpressionSeq())

    def parse_expression(self, expression_string):
        istream = InputStream(expression_string)
        parser = self.start_parser(istream)
        ev = ExpressionVisitor()
        return ev.visit(parser.expression())


class BatchVisitor(SqlSmallVisitor):
    def visitBatch(self, ctx):
        qv = QueryVisitor()
        queries = [q for q in [qv.visit(c) for c in ctx.children] if q is not None]
        return Batch(queries)


class QueryVisitor(SqlSmallVisitor):
    def visitQuery(self, ctx):

        # SELECT and FROM are required
        select = SelectVisitor().visit(ctx.selectClause())
        source = FromVisitor().visit(ctx.fromClause())

        wc = ctx.whereClause()
        where = WhereVisitor().visit(wc) if wc is not None else None

        hc = ctx.havingClause()
        having = HavingVisitor().visit(hc) if hc is not None else None

        ac = ctx.aggregationClause()
        agg = AggregateVisitor().visit(ac) if ac is not None else None

        oc = ctx.orderClause()
        order = OrderVisitor().visit(oc) if oc is not None else None

        limit = None
        if hasattr(ctx, "limitClause"):
            lc = ctx.limitClause()
            limit = LimitVisitor().visit(lc) if lc is not None else None

        return Query(select, source, where, agg, having, order, limit)


class SelectVisitor(SqlSmallVisitor):
    def visitSelectClause(self, ctx):
        nev = NamedExpressionVisitor()
        namedExpressions = nev.visit(ctx.namedExpressionSeq())

        sq = ctx.setQuantifier()
        tc = sq.topClause() if sq is not None else None
        if tc is None:
            quantifier = None if sq is None else Token(sq.getText())
        else:
            quantifier = LimitVisitor().visit(tc)

        return Select(quantifier, [ne for ne in namedExpressions if ne is not None])


class FromVisitor(SqlSmallVisitor):
    def visitFromClause(self, ctx):
        rv = RelationVisitor()
        relations = [rv.visit(rel) for rel in ctx.relation()]
        return From(relations)


class AggregateVisitor(SqlSmallVisitor):
    def visitAggregationClause(self, ctx):
        groups = ctx.groupingExpressions
        ev = ExpressionVisitor()
        cols = [GroupingExpression(ev.visit(g)) for g in groups]
        return Aggregate(cols)


class WhereVisitor(SqlSmallVisitor):
    def visitWhereClause(self, ctx):
        bev = BooleanExpressionVisitor()
        return Where(bev.visit(ctx.booleanExpression()))


class HavingVisitor(SqlSmallVisitor):
    def visitHavingClause(self, ctx):
        bev = BooleanExpressionVisitor()
        return Having(bev.visit(ctx.booleanExpression()))


class NamedExpressionVisitor(SqlSmallVisitor):
    def visitNamedExpressionSeq(self, ctx):
        return [self.visit(ne) for ne in ctx.namedExpression()]

    def visitNamedExpression(self, ctx):
        expression = ExpressionVisitor().visit(ctx.expression())
        name = Identifier(ctx.name.getText()) if ctx.name is not None else None
        return NamedExpression(name, expression)


class OrderVisitor(SqlSmallVisitor):
    def visitOrderClause(self, ctx):
        sortItems = [self.visit(si) for si in ctx.sortItem()]
        return Order(sortItems)

    def visitSortItem(self, ctx):
        ev = ExpressionVisitor()
        expr = ev.visit(ctx.expression())
        if ctx.DESC() is not None:
            o = Token("DESC")
        elif ctx.ASC() is not None:
            o = Token("ASC")
        else:
            o = None
        return SortItem(expr, o)


class LimitVisitor(SqlSmallVisitor):
    def visitLimitClause(self, ctx):
        return Limit(int(ctx.n.getText()))

    def visitTopClause(self, ctx):
        return Top(int(ctx.n.getText()))


class RelationVisitor(SqlSmallVisitor):
    def visitRelation(self, ctx):
        primary = self.visit(ctx.relationPrimary())
        jr = ctx.joinRelation()
        joins = [self.visit(j) for j in jr] if jr is not None else None
        return Relation(primary, joins)

    def visitTable(self, ctx):
        alias = Identifier(ctx.alias.getText()) if ctx.alias is not None else None
        return Table(Identifier(ctx.qualifiedTableName().getText()), alias)

    def visitAliasedQuery(self, ctx):
        alias = Identifier(ctx.alias.getText()) if ctx.alias is not None else None
        qv = QueryVisitor()
        return AliasedSubquery(qv.visitQuery(ctx.subquery()), alias)

    def visitAliasedRelation(self, ctx):
        alias = Identifier(ctx.alias.getText()) if ctx.alias is not None else None
        relation = self.visit(ctx.relation())
        return AliasedRelation(relation, alias)

    def visitJoinRelation(self, ctx):
        joinType = Token(allText(ctx.joinType()))
        right = RelationVisitor().visit(ctx.right)
        crit = ctx.joinCriteria()
        if type(crit) is SqlSmallParser.BooleanJoinContext:
            bev = BooleanExpressionVisitor()
            criteria = BooleanJoinCriteria(bev.visit(crit.booleanExpression()))
        elif type(crit) is SqlSmallParser.UsingJoinContext:
            ids = crit.identifier()
            criteria = UsingJoinCriteria([Column(i.getText()) for i in ids])
        else:
            criteria = None
        return Join(joinType, right, criteria)


class ExpressionVisitor(SqlSmallVisitor):
    def visitColumnName(self, ctx):
        return Column(ctx.name.getText())

    def visitCaseExpr(self, ctx):
        return CaseExpressionVisitor().visit(ctx)

    def visitAllExpr(self, ctx):
        ident = ctx.allExpression().identifier()
        return AllColumns(ident.getText() if ident is not None else None)

    def visitMultiply(self, ctx):
        return ArithmeticExpression(self.visit(ctx.left), Op("*"), self.visit(ctx.right))

    def visitDivide(self, ctx):
        return ArithmeticExpression(self.visit(ctx.left), Op("/"), self.visit(ctx.right))

    def visitModulo(self, ctx):
        return ArithmeticExpression(self.visit(ctx.left), Op("%"), self.visit(ctx.right))

    def visitAdd(self, ctx):
        return ArithmeticExpression(self.visit(ctx.left), Op("+"), self.visit(ctx.right))

    def visitSubtract(self, ctx):
        return ArithmeticExpression(self.visit(ctx.left), Op("-"), self.visit(ctx.right))

    def visitDecimalLiteral(self, ctx):
        return Literal(float(allText(ctx)))

    def visitIntegerLiteral(self, ctx):
        return Literal(int(allText(ctx)))

    def visitStringLiteral(self, ctx):
        text = str(allText(ctx))
        t_len = len(text)
        value = text
        if t_len > 1:
            l_delim = text[0]
            r_delim = text[t_len - 1]
            if r_delim == "'" and l_delim == "'":
                # this is the expected case for all stringLiteral
                value = text[1:t_len - 1]
        return Literal(value, text)

    def visitTrueLiteral(self, ctx):
        return Literal(True)

    def visitFalseLiteral(self, ctx):
        return Literal(False)

    def visitNullLiteral(self, ctx):
        return Literal(None)

    def visitAggFunc(self, ctx):
        fname = FuncName(ctx.function.getText().upper())
        qt = ctx.setQuantifier()
        quantifier = Token(qt.getText().upper()) if qt is not None else None
        return AggFunction(fname, quantifier, self.visit(ctx.expression()))

    def visitSubqueryExpr(self, ctx):
        esq = ctx.expressionSubquery()
        sq = esq.subquery()
        qv = QueryVisitor()
        q = qv.visitQuery(sq)
        return AliasedSubquery(q, None)

    def visitNestedExpr(self, ctx):
        return NestedExpression(self.visit(ctx.expression()))

    def visitIifFunc(self, ctx):
        test = BooleanExpressionVisitor().visit(ctx.test)
        yes = ExpressionVisitor().visit(ctx.yes)
        no = ExpressionVisitor().visit(ctx.no)
        return IIFFunction(test, yes, no)

    def visitRoundFunction(self, ctx):
        expression = ExpressionVisitor().visit(ctx.expression())
        digits = ExpressionVisitor().visit(ctx.digits)
        return RoundFunction(expression, digits)

    def visitMathFunc(self, ctx):
        fname = FuncName(ctx.function.getText().upper())
        return MathFunction(fname, self.visit(ctx.expression()))

    def visitChooseFunc(self, ctx):
        expression = ExpressionVisitor().visit(ctx.index)
        choices = Seq([ExpressionVisitor().visit(e) for e in ctx.literal()])
        return ChooseFunction(expression, choices)

    def visitPowerFunction(self, ctx):
        return PowerFunction(
            ExpressionVisitor().visit(ctx.expression()), ExpressionVisitor().visit(ctx.number())
        )

    def visitBareFunction(self, ctx):
        return BareFunction(FuncName(ctx.function.getText().upper()))

    def visitRankingFunction(self, ctx):
        fname = FuncName(ctx.function.getText().upper())
        over = self.visit(ctx.overClause())
        return RankingFunction(fname, over)

    def visitOverClause(self, ctx):
        partition = (
            ExpressionVisitor().visit(ctx.expression()) if ctx.expression() is not None else None
        )
        oc = ctx.orderClause()
        order = OrderVisitor().visit(oc) if oc is not None else None
        return OverClause(partition, order)


class CaseExpressionVisitor(SqlSmallVisitor):
    def visitCaseBaseExpr(self, ctx):
        wxp = ctx.whenBaseExpression()
        whenExpressions = [self.visit(we) for we in wxp]
        expression = ExpressionVisitor().visit(ctx.baseCaseExpr)
        else_expr = ExpressionVisitor().visit(ctx.elseExpr) if ctx.elseExpr is not None else None
        return CaseExpression(expression, whenExpressions, else_expr)

    def visitCaseWhenExpr(self, ctx):
        wxp = ctx.whenExpression()
        whenExpressions = [self.visit(we) for we in wxp]
        expression = None
        else_expr = ExpressionVisitor().visit(ctx.elseExpr) if ctx.elseExpr is not None else None
        return CaseExpression(expression, whenExpressions, else_expr)

    def visitWhenExpression(self, ctx):
        expression = BooleanExpressionVisitor().visit(ctx.baseBoolExpr)
        thenExpression = ExpressionVisitor().visit(ctx.thenExpr)
        return WhenExpression(expression, thenExpression)

    def visitWhenBaseExpression(self, ctx):
        expression = ExpressionVisitor().visit(ctx.baseWhenExpr)
        thenExpression = ExpressionVisitor().visit(ctx.thenExpr)
        return WhenExpression(expression, thenExpression)


class BooleanExpressionVisitor(SqlSmallVisitor):
    def visitLogicalNot(self, ctx):
        return LogicalNot(self.visit(ctx.booleanExpression()))

    def visitComparison(self, ctx):
        ev = ExpressionVisitor()
        return BooleanCompare(ev.visit(ctx.left), Op(ctx.op.getText()), ev.visit(ctx.right))

    def visitConjunction(self, ctx):
        return BooleanCompare(self.visit(ctx.left), Op("AND"), self.visit(ctx.right))

    def visitDisjunction(self, ctx):
        return BooleanCompare(self.visit(ctx.left), Op("OR"), self.visit(ctx.right))

    def visitNestedBoolean(self, ctx):
        return NestedBoolean(self.visit(ctx.booleanExpression()))

    def visitPredicated(self, ctx):
        expression = ExpressionVisitor().visit(ctx.expression())
        predicate = self.visit(ctx.predicate())
        return PredicatedExpression(expression, predicate)

    def visitInCondition(self, ctx):
        is_not = ctx.NOT() is not None
        expressions = Seq([ExpressionVisitor().visit(e) for e in ctx.expression()])
        return InCondition(expressions, is_not)

    def visitIsCondition(self, ctx):
        is_not = ctx.NOT() is not None
        if ctx.TRUE() is not None:
            value = Literal(True)
        elif ctx.FALSE() is not None:
            value = Literal(False)
        elif ctx.NULL() is not None:
            value = Literal(None)
        else:
            raise ValueError("Unknown condition in IS clause: " + allText(ctx))
        return IsCondition(value, is_not)

    def visitBetweenCondition(self, ctx):
        is_not = ctx.NOT() is not None
        lower = ExpressionVisitor().visit(ctx.lower)
        upper = ExpressionVisitor().visit(ctx.upper)
        return BetweenCondition(lower, upper, is_not)

    def visitQualifiedColumnName(self, ctx):
        return ColumnBoolean(Column(ctx.getText()))


def allText(ctx):
    """
        This method is used to grab text with whitespace
        for a terminal node of the AST that hasn't been
        strongly-typed yet.  Should not be used for lexer
        tokens, and should be replaced over time.
    """
    a = ctx.start.start
    b = ctx.stop.stop
    inp = ctx.start.getInputStream()
    return inp.getText(a, b)
