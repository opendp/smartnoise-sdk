from typing import Union, Sequence, Iterable

from antlr4.atn.Transition import AtomTransition

def flatten(lst):
    for item in lst:
        if isinstance(item,Iterable) and not isinstance(item,str):
            yield from flatten(item)
        else:
            yield item

class StringLiteral:
    def __init__(self, value : str):
        self.value = value

class NumericLiteral:
    def __init__(self, value : Union[int, float]):
        self.value = value

class NullLiteral:
    def __init__(self):
        pass

class Identifier:
    def __init__(self, name : str):
        self.name = name

class AllNodes:
    def evaluate(self, node):
        return [c for c in node.children() if c is not None]

class AllAttributes:
    def evaluate(self, node):
        d = node.__dict__
        return [Attribute(k, d[k]) for k in d.keys() if d[k] is not None]

class Attribute:
    def __init__(self, name : str, value : str = None):
        self.name = name
        self.value = value

class Statement:
    def __init__(self, steps : Sequence):
        self.steps = steps
    def evaluate(self, node):
        n = [node]
        for s in self.steps:
            res = list(flatten([s.evaluate(nn) for nn in n]))
            n = [r for r in res if r is not None]
        return [nn for nn in n if nn is not None]

class Condition:
    def __init__(self, left : Statement, operator : str, right : Statement):
        self.left = left
        self.right = right
        self.operator = operator
    def evaluate(self, node):
        l = self.left.evaluate(node)
        if self.operator is None:
            return len(l) >= 1
        else:
            raise ValueError("Not implemented")

class RootSelect:
    def __init__(self, target : Union[Identifier, Attribute], condition : Condition = None):
        self.target = target
        self.condition = condition
    def evaluate(self, node):
        r = []
        if isinstance(self.target, Identifier):
            cname = node.__class__.__name__
            if self.target.name == cname:
                r.append(node)
        elif isinstance(self.target, Attribute): # it's an attribute
            if hasattr(node, self.target.name):
                v = getattr(node, self.target.name)
                if v is not None:
                    r.append(Attribute(self.target.name, v))
        if r != [] and self.condition is not None:
            if not self.condition.evaluate(node):
                return []
        return r

class ChildSelect:
    def __init__(self, target : Union[Identifier, Attribute], condition : Condition = None):
        self.target = target
        self.condition = condition
    def evaluate(self, node):
        r = []
        if isinstance(self.target, Identifier):
            for n in node.children():
                cname = n.__class__.__name__
                if self.target.name == cname:
                    r.append(n)
        elif isinstance(self.target, Attribute): # it's an attribute
            if hasattr(node, self.target.name):
                v = getattr(node, self.target.name)
                if v is not None:
                    r.append(Attribute(self.target.name, v))
        elif isinstance(self.target, (AllNodes, AllAttributes)):
            r.append(list(flatten(self.target.evaluate(node))))
        if r != [] and self.condition is not None:
            r = [n for n in r if self.condition.evaluate(n)]
        return r

class DescendantSelect:
    # DescendantSelector over child nodes
    def __init__(self, target : Union[Identifier, Attribute], condition : Condition = None):
        self.target = target
        self.condition = condition
    def evaluate(self, node):
        r = []
        # recurse descendants
        step = RootDescendantSelect(self.target, self.condition) # should condition here be None?
        for n in node.children():
            if n is not None:
                r.append(list(flatten(step.evaluate(n))))
        if r != [] and self.condition is not None:
            r = [n for n in r if self.condition.evaluate(n)]
        return flatten(r)

class RootDescendantSelect:
    def __init__(self, target : Union[Identifier, Attribute], condition : Condition = None):
        self.target = target
        self.condition = condition
    def evaluate(self, node):
        r = []
        # first look at self
        if isinstance(self.target, Identifier):
            cname = node.__class__.__name__
            if self.target.name == cname:
                r.append(node)
        elif isinstance(self.target, Attribute): # it's an attribute
            if hasattr(node, self.target.name):
                v = getattr(node, self.target.name)
                if v is not None:
                    r.append(Attribute(self.target.name, v))
        elif isinstance(self.target, (AllNodes, AllAttributes)):
            r.append(list(flatten(self.target.evaluate(node))))
        # recurse descendants
        for n in node.children():
            if n is not None:
                r.append(list(flatten(self.evaluate(n))))
        r = list(flatten(r))
        if r != [] and self.condition is not None:
            r = [n for n in r if self.condition.evaluate(n)]
        return list(flatten(r))
