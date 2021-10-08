
from antlr4.tree.Tree import TerminalNodeImpl
from snsql.xpath.parser.XPathLexer import XPathLexer  # type: ignore
from snsql.xpath.parser.XPathParser import XPathParser  # type: ignore
from snsql.xpath.parser.XPathVisitor import XPathVisitor  # type: ignore
from snsql.xpath.parser.XPathErrorListener import SyntaxErrorListener  # type: ignore

from antlr4 import *  # type: ignore
from snsql.xpath.ast import *


class XPath:
    def start_parser(self, stream):
        lexer = XPathLexer(stream)
        stream = CommonTokenStream(lexer)
        parser = XPathParser(stream)
        parser._interp.predictionMode = PredictionMode.LL_EXACT_AMBIG_DETECTION
        lexer._listeners = [SyntaxErrorListener(), DiagnosticErrorListener()]
        parser._listeners = [SyntaxErrorListener(), DiagnosticErrorListener()]
        return parser

    def parse(self, statement):
        istream = InputStream(statement)
        parser = self.start_parser(istream)
        v = StatementVisitor()
        s = v.visit(parser.statement())
        return s

    def parse_only(self, statement):
        istream = InputStream(statement)
        parser = self.start_parser(istream)
        XPathVisitor().visit(parser.statement())
        return None


class StatementVisitor(XPathVisitor):
    def visitStatement(self, ctx):
        return self.visit(ctx.innerStatement())
    def visitInnerStatement(self, ctx: XPathParser.InnerStatementContext):
        v = SelectorVisitor()
        return Statement([v.visit(step) for step in ctx.children if not isinstance(step, TerminalNodeImpl)])

class SelectorVisitor(XPathVisitor):
    def getIdentifier(self, ctx):
        target = None
        if ctx.ident is not None:
            target = Identifier(ctx.ident.text)
        elif ctx.attr is not None:
            txt = ctx.attr.text.replace('@', '')
            target = Attribute(txt)
        elif ctx.allsel is not None:
            sel = ctx.allsel.getText()
            if sel == '@*':
                target=AllAttributes()
            else:
                target=AllNodes()
        return target
    def getBoolean(self, ctx):
        bv = BooleanVisitor()
        condition = ctx.booleanSelector()
        if condition is None:
            return None
        else:
            return bv.visit(condition)

    def visitChildSelector(self, ctx: XPathParser.ChildSelectorContext):
        target =self.getIdentifier(ctx)
        b = self.getBoolean(ctx)
        return ChildSelect(target, b)
    def visitRootSelector(self, ctx: XPathParser.RootSelectorContext):
        target =self.getIdentifier(ctx)
        b = self.getBoolean(ctx)
        return RootSelect(target, b)
    def visitRootDescendantSelector(self, ctx: XPathParser.RootDescendantSelectorContext):
        target =self.getIdentifier(ctx)
        b = self.getBoolean(ctx)
        return RootDescendantSelect(target, b)
    def visitDescendantSelector(self, ctx: XPathParser.DescendantSelectorContext):
        target =self.getIdentifier(ctx)
        b = self.getBoolean(ctx)
        return DescendantSelect(target, b)
    def visitIndexSelector(self, ctx: XPathParser.IndexSelectorContext):
        return IndexSelector(int(ctx.index.text))

class BooleanVisitor(XPathVisitor):
    def visitBooleanSelector(self, ctx: XPathParser.BooleanSelectorContext):
        sv = StatementVisitor()
        lv = LiteralVisitor()
        if ctx.left is not None:
            left = sv.visit(ctx.left)
        elif ctx.llit is not None:
            left = lv.visit(ctx.llit)
        op = None if ctx.op is None else lv.visit(ctx.op)
        stmt = None if ctx.stmt is None else sv.visit(ctx.stmt)
        lit = None if ctx.rlit is None else lv.visit(ctx.rlit)
        if lit is not None:
            stmt = lit
        return Condition(left, op, stmt)

class LiteralVisitor(XPathVisitor):
    def visitNullLiteral(self, ctx: XPathParser.NullLiteralContext):
        return NullLiteral()
    def visitDecimalLiteral(self, ctx: XPathParser.DecimalLiteralContext):
        return NumericLiteral(float(ctx.getText()))
    def visitIntegerLiteral(self, ctx: XPathParser.IntegerLiteralContext):
        return NumericLiteral(int(ctx.getText()))
    def visitStringLiteral(self, ctx: XPathParser.StringLiteralContext):
        return StringLiteral(ctx.getText()[1:-1])
    def visitComparisonOperator(self, ctx: XPathParser.ComparisonOperatorContext):
        return ctx.getText()


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
