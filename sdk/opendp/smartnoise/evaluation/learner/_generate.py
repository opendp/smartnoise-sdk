"""
    Simple generator from Context-free Grammar
    Supports quantifiers *, +, ?, {n}, and {m,n}
"""

import numpy as np
import string
from itertools import chain
import random

# random.seed(0)


class Token(object):
    def __init__(self, token, children):
        self.token = token
        self.quantifier = Grammar.is_quantifier(token)
        self.literal = Grammar.is_literal(token)
        self.children = (
            None
            if children is None
            else [c for c in children if c.token != "" and c.token != "null"]
        )

    def __str__(self):
        if self.literal:
            return self.token.replace("'", "")
        elif self.quantifier:
            return ", ".join(s for s in [str(c) for c in self.children] if s != "")
        elif self.token == "collapse":
            return (
                "".join(s for s in [str(c) for c in self.children] if s != "")
                .replace(" ", "")
                .replace(",", "")
            )
        else:
            return " ".join(s for s in [str(c) for c in self.children] if s != "")


class Grammar(object):
    def __init__(self, numofquery):
        self.grammar = {"null": [""]}
        self.numofquery = numofquery

    @staticmethod
    def validate_load(token):
        if token.startswith("'"):
            raise Exception("Literals not allowed as token names: {0}".format(token))
        if Grammar.is_quantifier(token):
            raise Exception("Quantifiers not allowed in token names: {0}".format(token))
        if Grammar.is_builtin(token):
            raise Exception("Attempting to define a token that is built-in: {0}".format(token))
        return True

    @staticmethod
    def is_quantifier(token):
        return any([token.endswith(q) for q in ["*", "?", "+", "}"]])

    @staticmethod
    def is_builtin(token):
        return (
            any(
                [
                    token.startswith(q)
                    for q in ["rand_int(", "rand_float(", "rand_string(", "collapse("]
                ]
            )
            or token == "null"
        )

    @staticmethod
    def is_literal(token):
        return True if token.startswith("'") else False

    @staticmethod
    def make_literal(token):
        return Token("'" + str(token) + "'", None)

    @staticmethod
    def rand_int(m, n):
        return str(np.random.random_integers(m, n))

    @staticmethod
    def rand_float(m, n):
        return np.random.uniform(m, n)

    @staticmethod
    def val_between(token, left, right):
        # no checks here; this will throw index error if left and right don't match
        return token.split(left)[1].split(right)[0]

    @staticmethod
    def vals_between(token, left, right):
        # no checks here; this will throw index error if left and right don't match
        return [s.strip() for s in Grammar.val_between(token, left, right).split(",")]

    @staticmethod
    def quantifier_definition(quant):
        if quant.endswith("*"):
            bare = quant.replace("*", "")
            return [[bare], ["null"], [bare, quant]]
        if quant.endswith("?"):
            bare = quant.replace("?", "")
            return [[bare], ["null"]]
        if quant.endswith("+"):
            bare = quant.replace("+", "")
            return [[bare], [bare, bare]]
        if quant.endswith("}"):
            bare = quant.split("{")[0]
            vals = [int(sval) for sval in Grammar.vals_between(quant, "{", "}")]
            min_ = vals[0]
            max_ = min_ if len(vals) == 1 else vals[1]
            return [[bare for i in range(s)] for s in range(min_, max_ + 1)]

    def generate_builtin(self, token):
        bare = token.split("(")[0]
        if bare == "null":
            return Token(bare, None)
        if bare == "collapse":
            val = Grammar.val_between(token, "(", ")")
            return Token(bare, [self.generate(val)])
        elif bare == "rand_float":
            vals = Grammar.vals_between(token, "(", ")")
            m, n = [float(v) for v in vals]
            return Token(bare, [Grammar.make_literal(Grammar.rand_float(m, n))])
        elif bare == "rand_int":
            vals = Grammar.vals_between(token, "(", ")")
            m, n = [int(v) for v in vals]
            return Token(bare, [Grammar.make_literal(Grammar.rand_int(m, n))])
        elif bare == "rand_string":
            val = Grammar.val_between(token, "(", ")")
            return Grammar.make_literal(
                '"'
                + "".join([np.random.choice(list(string.ascii_letters)) for i in range(int(val))])
                + '"'
            )
        else:
            raise ValueError("Attempt to call unknown built-in: {0}".format(token))

    def load(self, rules):
        grammar = self.grammar
        for line in rules:
            if line.strip() == "" or line.strip().startswith("#"):
                continue
            token, prod = [p.strip() for p in line.split("->")]
            self.validate_load(token)
            prods = [p.strip() for p in prod.split("|")]
            tokens = [[t.strip() for t in prod.split(" ")] for prod in prods]
            for quant in [
                t
                for t in chain.from_iterable(tokens)
                if self.is_quantifier(t) and (t not in grammar)
            ]:
                grammar[quant] = self.quantifier_definition(quant)
            if token in grammar:
                grammar[token].extend(tokens)
            else:
                grammar[token] = tokens
        return grammar

    def generate(self, token):
        grammar = self.grammar
        if Grammar.is_literal(token):
            return Token(token, None)  # terminal node
        if Grammar.is_builtin(token):
            return self.generate_builtin(token)
        if self.is_quantifier(token) and (token not in grammar):
            grammar[token] = self.quantifier_definition(token)
        prods = grammar[token]
        prod = prods[np.random.choice(len(prods))]
        return Token(token, [self.generate(t) for t in prod])
