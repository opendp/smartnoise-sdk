from antlr4 import *

class SyntaxErrorListener():
    def syntaxError(self, recognizer, offendingToken, line, column, msg, e):
        if offendingToken is None:
            raise ValueError("Lexer error at line {0} column {1}.  Message: {2}".format(line, column, msg))
        elif recognizer.symbolicNames[offendingToken.type] == "UNSUPPORTED":
            raise ValueError("Reserved SQL keyword is unsupported in this parser: {0} at line {1} column {2}.  Message: {3}".format(offendingToken.text, line, column, msg))
        else:
            raise ValueError("Bad token {0} at line {1} column {2}.  Message: {3}".format(offendingToken.text, line, column, msg))
    def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs):
        # use DiagnosticErrorListener() to get full diagnostics
        # use this stub to raise ValueError if needed for unit tests to throw specific ambiguity errors
        pass
    def reportAttemptingFullContext(self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs):
        # use DiagnosticErrorListener() to get full diagnostics
        # use this stub to raise ValueError if needed for unit tests to throw specific ambiguity errors
        # raise ValueError("Attempting Full Context")
        pass
    def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
        # use DiagnosticErrorListener() to get full diagnostics
        # use this stub to raise ValueError if needed for unit tests to throw specific ambiguity errors
        # raise ValueError("Found Exact")
        pass