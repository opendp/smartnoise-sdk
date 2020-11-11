# Generated from SqlSmall.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .SqlSmallParser import SqlSmallParser
else:
    from SqlSmallParser import SqlSmallParser

# This class defines a complete generic visitor for a parse tree produced by SqlSmallParser.

class SqlSmallVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by SqlSmallParser#batch.
    def visitBatch(self, ctx:SqlSmallParser.BatchContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#query.
    def visitQuery(self, ctx:SqlSmallParser.QueryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#subquery.
    def visitSubquery(self, ctx:SqlSmallParser.SubqueryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#expressionSubquery.
    def visitExpressionSubquery(self, ctx:SqlSmallParser.ExpressionSubqueryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#selectClause.
    def visitSelectClause(self, ctx:SqlSmallParser.SelectClauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#fromClause.
    def visitFromClause(self, ctx:SqlSmallParser.FromClauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#whereClause.
    def visitWhereClause(self, ctx:SqlSmallParser.WhereClauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#aggregationClause.
    def visitAggregationClause(self, ctx:SqlSmallParser.AggregationClauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#havingClause.
    def visitHavingClause(self, ctx:SqlSmallParser.HavingClauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#orderClause.
    def visitOrderClause(self, ctx:SqlSmallParser.OrderClauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#limitClause.
    def visitLimitClause(self, ctx:SqlSmallParser.LimitClauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#topClause.
    def visitTopClause(self, ctx:SqlSmallParser.TopClauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#joinRelation.
    def visitJoinRelation(self, ctx:SqlSmallParser.JoinRelationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#joinType.
    def visitJoinType(self, ctx:SqlSmallParser.JoinTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#booleanJoin.
    def visitBooleanJoin(self, ctx:SqlSmallParser.BooleanJoinContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#usingJoin.
    def visitUsingJoin(self, ctx:SqlSmallParser.UsingJoinContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#sortItem.
    def visitSortItem(self, ctx:SqlSmallParser.SortItemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#setQuantifier.
    def visitSetQuantifier(self, ctx:SqlSmallParser.SetQuantifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#relation.
    def visitRelation(self, ctx:SqlSmallParser.RelationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#table.
    def visitTable(self, ctx:SqlSmallParser.TableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#aliasedQuery.
    def visitAliasedQuery(self, ctx:SqlSmallParser.AliasedQueryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#aliasedRelation.
    def visitAliasedRelation(self, ctx:SqlSmallParser.AliasedRelationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#caseBaseExpr.
    def visitCaseBaseExpr(self, ctx:SqlSmallParser.CaseBaseExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#caseWhenExpr.
    def visitCaseWhenExpr(self, ctx:SqlSmallParser.CaseWhenExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#namedExpression.
    def visitNamedExpression(self, ctx:SqlSmallParser.NamedExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#namedExpressionSeq.
    def visitNamedExpressionSeq(self, ctx:SqlSmallParser.NamedExpressionSeqContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#whenExpression.
    def visitWhenExpression(self, ctx:SqlSmallParser.WhenExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#whenBaseExpression.
    def visitWhenBaseExpression(self, ctx:SqlSmallParser.WhenBaseExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#add.
    def visitAdd(self, ctx:SqlSmallParser.AddContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#subtract.
    def visitSubtract(self, ctx:SqlSmallParser.SubtractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#nestedExpr.
    def visitNestedExpr(self, ctx:SqlSmallParser.NestedExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#subqueryExpr.
    def visitSubqueryExpr(self, ctx:SqlSmallParser.SubqueryExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#allExpr.
    def visitAllExpr(self, ctx:SqlSmallParser.AllExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#functionExpr.
    def visitFunctionExpr(self, ctx:SqlSmallParser.FunctionExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#rankFunction.
    def visitRankFunction(self, ctx:SqlSmallParser.RankFunctionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#literalExpr.
    def visitLiteralExpr(self, ctx:SqlSmallParser.LiteralExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#divide.
    def visitDivide(self, ctx:SqlSmallParser.DivideContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#caseExpr.
    def visitCaseExpr(self, ctx:SqlSmallParser.CaseExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#multiply.
    def visitMultiply(self, ctx:SqlSmallParser.MultiplyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#modulo.
    def visitModulo(self, ctx:SqlSmallParser.ModuloContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#columnName.
    def visitColumnName(self, ctx:SqlSmallParser.ColumnNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#betweenCondition.
    def visitBetweenCondition(self, ctx:SqlSmallParser.BetweenConditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#inCondition.
    def visitInCondition(self, ctx:SqlSmallParser.InConditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#isCondition.
    def visitIsCondition(self, ctx:SqlSmallParser.IsConditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#bareFunc.
    def visitBareFunc(self, ctx:SqlSmallParser.BareFuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#roundFunc.
    def visitRoundFunc(self, ctx:SqlSmallParser.RoundFuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#powerFunc.
    def visitPowerFunc(self, ctx:SqlSmallParser.PowerFuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#aggFunc.
    def visitAggFunc(self, ctx:SqlSmallParser.AggFuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#mathFunc.
    def visitMathFunc(self, ctx:SqlSmallParser.MathFuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#iifFunc.
    def visitIifFunc(self, ctx:SqlSmallParser.IifFuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#chooseFunc.
    def visitChooseFunc(self, ctx:SqlSmallParser.ChooseFuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#boolColumn.
    def visitBoolColumn(self, ctx:SqlSmallParser.BoolColumnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#logicalNot.
    def visitLogicalNot(self, ctx:SqlSmallParser.LogicalNotContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#comparison.
    def visitComparison(self, ctx:SqlSmallParser.ComparisonContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#predicated.
    def visitPredicated(self, ctx:SqlSmallParser.PredicatedContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#conjunction.
    def visitConjunction(self, ctx:SqlSmallParser.ConjunctionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#disjunction.
    def visitDisjunction(self, ctx:SqlSmallParser.DisjunctionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#nestedBoolean.
    def visitNestedBoolean(self, ctx:SqlSmallParser.NestedBooleanContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#bareFunction.
    def visitBareFunction(self, ctx:SqlSmallParser.BareFunctionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#rankingFunction.
    def visitRankingFunction(self, ctx:SqlSmallParser.RankingFunctionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#roundFunction.
    def visitRoundFunction(self, ctx:SqlSmallParser.RoundFunctionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#powerFunction.
    def visitPowerFunction(self, ctx:SqlSmallParser.PowerFunctionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#comparisonOperator.
    def visitComparisonOperator(self, ctx:SqlSmallParser.ComparisonOperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#booleanValue.
    def visitBooleanValue(self, ctx:SqlSmallParser.BooleanValueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#allExpression.
    def visitAllExpression(self, ctx:SqlSmallParser.AllExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#stringLiteral.
    def visitStringLiteral(self, ctx:SqlSmallParser.StringLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#numberLiteral.
    def visitNumberLiteral(self, ctx:SqlSmallParser.NumberLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#trueLiteral.
    def visitTrueLiteral(self, ctx:SqlSmallParser.TrueLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#falseLiteral.
    def visitFalseLiteral(self, ctx:SqlSmallParser.FalseLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#nullLiteral.
    def visitNullLiteral(self, ctx:SqlSmallParser.NullLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#rankingFunctionName.
    def visitRankingFunctionName(self, ctx:SqlSmallParser.RankingFunctionNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#aggregateFunctionName.
    def visitAggregateFunctionName(self, ctx:SqlSmallParser.AggregateFunctionNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#mathFunctionName.
    def visitMathFunctionName(self, ctx:SqlSmallParser.MathFunctionNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#bareFunctionName.
    def visitBareFunctionName(self, ctx:SqlSmallParser.BareFunctionNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#overClause.
    def visitOverClause(self, ctx:SqlSmallParser.OverClauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#aliasedSubquery.
    def visitAliasedSubquery(self, ctx:SqlSmallParser.AliasedSubqueryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#aliasedTableOrSubquerySeq.
    def visitAliasedTableOrSubquerySeq(self, ctx:SqlSmallParser.AliasedTableOrSubquerySeqContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#aliasedTableSeq.
    def visitAliasedTableSeq(self, ctx:SqlSmallParser.AliasedTableSeqContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#aliasedTableName.
    def visitAliasedTableName(self, ctx:SqlSmallParser.AliasedTableNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#qualifiedTableName.
    def visitQualifiedTableName(self, ctx:SqlSmallParser.QualifiedTableNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#qualifiedColumnName.
    def visitQualifiedColumnName(self, ctx:SqlSmallParser.QualifiedColumnNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#identifier.
    def visitIdentifier(self, ctx:SqlSmallParser.IdentifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#decimalLiteral.
    def visitDecimalLiteral(self, ctx:SqlSmallParser.DecimalLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SqlSmallParser#integerLiteral.
    def visitIntegerLiteral(self, ctx:SqlSmallParser.IntegerLiteralContext):
        return self.visitChildren(ctx)



del SqlSmallParser