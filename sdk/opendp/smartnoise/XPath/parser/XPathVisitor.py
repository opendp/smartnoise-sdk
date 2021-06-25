# Generated from XPath.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .XPathParser import XPathParser
else:
    from XPathParser import XPathParser

# This class defines a complete generic visitor for a parse tree produced by XPathParser.

class XPathVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by XPathParser#statement.
    def visitStatement(self, ctx:XPathParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#innerStatement.
    def visitInnerStatement(self, ctx:XPathParser.InnerStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#childSelector.
    def visitChildSelector(self, ctx:XPathParser.ChildSelectorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#rootSelector.
    def visitRootSelector(self, ctx:XPathParser.RootSelectorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#rootDescendantSelector.
    def visitRootDescendantSelector(self, ctx:XPathParser.RootDescendantSelectorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#descendantSelector.
    def visitDescendantSelector(self, ctx:XPathParser.DescendantSelectorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#booleanSelector.
    def visitBooleanSelector(self, ctx:XPathParser.BooleanSelectorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#indexSelector.
    def visitIndexSelector(self, ctx:XPathParser.IndexSelectorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#allSelect.
    def visitAllSelect(self, ctx:XPathParser.AllSelectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#allNodes.
    def visitAllNodes(self, ctx:XPathParser.AllNodesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#allAttributes.
    def visitAllAttributes(self, ctx:XPathParser.AllAttributesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#comparisonOperator.
    def visitComparisonOperator(self, ctx:XPathParser.ComparisonOperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#booleanValue.
    def visitBooleanValue(self, ctx:XPathParser.BooleanValueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#allExpression.
    def visitAllExpression(self, ctx:XPathParser.AllExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#stringLiteral.
    def visitStringLiteral(self, ctx:XPathParser.StringLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#numberLiteral.
    def visitNumberLiteral(self, ctx:XPathParser.NumberLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#trueLiteral.
    def visitTrueLiteral(self, ctx:XPathParser.TrueLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#falseLiteral.
    def visitFalseLiteral(self, ctx:XPathParser.FalseLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#nullLiteral.
    def visitNullLiteral(self, ctx:XPathParser.NullLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#decimalLiteral.
    def visitDecimalLiteral(self, ctx:XPathParser.DecimalLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XPathParser#integerLiteral.
    def visitIntegerLiteral(self, ctx:XPathParser.IntegerLiteralContext):
        return self.visitChildren(ctx)



del XPathParser