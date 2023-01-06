from snsql.reader.base import Reader
from snsql.sql.reader.engine import Engine
import importlib

from snsql.sql.reader.probe import Probe

class SqlReader(Reader):
    @classmethod
    def get_reader_class(cls, engine):
        prefix = ""
        for eng in Engine.known_engines:
            if str(eng).lower() == engine.lower():
                prefix = str(eng)
        if prefix == "":
            return SqlReader()  # should this throw?
        else:
            mod_path = f"snsql.sql.reader.{Engine.class_map[prefix]}"
            module = importlib.import_module(mod_path)
            class_ = getattr(module, f"{prefix}Reader")
            return class_
    @classmethod
    def from_connection(cls, conn, engine=None, **kwargs):
        if engine is None:
            probe = Probe()
            engine = probe.engine(conn)
            if engine is None:
                raise ValueError("Unable to detect the database engine.  Please pass in engine parameter")
        _reader = cls.get_reader_class(engine)
        return _reader(conn=conn, **kwargs)
    def __init__(self, engine=None):
        self.compare = NameCompare.get_name_compare(engine)
        self.serializer = Serializer.get_serializer(engine)

    def execute(self, query, *ignore, accuracy:bool=False):
        raise NotImplementedError("Execute must be implemented on the inherited class")
    def _execute_ast(self, query, *ignore, accuracy:bool=False):
        if isinstance(query, str):
            raise ValueError("Please pass ASTs to execute_ast.  To execute strings, use execute.")
        if hasattr(self, "serializer") and self.serializer is not None:
            query_string = self.serializer.serialize(query)
        else:
            query_string = str(query)
        return self.execute(query_string, accuracy=accuracy)
    def _execute_ast_df(self, query, *ignore, accuracy:bool=False):
        return self._to_df(self._execute_ast(query, accuracy=accuracy))

"""
    Implements engine-specific identifier matching rules
    for escaped identifiers.
"""
class NameCompare:
    @classmethod
    def get_name_compare(cls, engine):
        prefix = ""
        for eng in Engine.known_engines:
            if str(eng).lower() == engine.lower():
                prefix = str(eng)
        if prefix == "":
            return NameCompare()
        else:
            mod_path = f"snsql.sql.reader.{Engine.class_map[prefix]}"
            module = importlib.import_module(mod_path)
            class_ = getattr(module, f"{prefix}NameCompare")
            return class_()

    def __init__(self, search_path=None):
        self.search_path = search_path if search_path is not None else []

    """
        True if schema portion of identifier used in query
        matches schema or metadata object.  Follows search
        path.  Pass in only the schema part.
    """

    def reserved(self):
        return ["select", "group", "on"]

    def schema_match(self, from_query, from_meta):
        if from_query.strip() == "" and from_meta in self.search_path:
            return True
        if from_meta.strip() == "" and from_query in self.search_path:
            return True
        return self.identifier_match(from_query, from_meta)

    """
        Uses database engine matching rules to report True
        if identifier used in query matches identifier
        of metadata object.  Pass in one part at a time.
    """
    def identifier_match(self, from_query, from_meta):
        return from_query == from_meta

    """
        Removes all escaping characters, keeping identifiers unchanged
    """
    def strip_escapes(self, value):
        return value.replace('"', "").replace("`", "").replace("[", "").replace("]", "")

    """
        True if any part of identifier is escaped
    """
    def is_escaped(self, identifier):
        return any([p[0] in ['"', "[", "`"] for p in identifier.split(".") if p != ""])

    """
        Converts proprietary escaping to SQL-92.  Supports multi-part identifiers
    """
    def clean_escape(self, identifier):
        escaped = []
        for p in identifier.split("."):
            if self.is_escaped(p):
                escaped.append(p.replace("[", '"').replace("]", '"').replace("`", '"'))
            else:
                escaped.append(p.lower())
        return ".".join(escaped)

    """
        Returns true if an identifier should
        be escaped.  Checks only one part per call.
    """
    def should_escape(self, identifier):
        if self.is_escaped(identifier):
            return False
        if identifier.lower() in self.reserved():
            return True
        if identifier.lower().replace(" ", "") == identifier.lower():
            return False
        else:
            return True

class Serializer:
    @classmethod
    def get_serializer(cls, engine):
        prefix = ""
        for eng in Engine.known_engines:
            if str(eng).lower() == engine.lower():
                prefix = str(eng)
        if prefix == "":
            return Serializer()
        else:
            mod_path = f"snsql.sql.reader.{Engine.class_map[prefix]}"
            module = importlib.import_module(mod_path)
            class_ = getattr(module, f"{prefix}Serializer")
            return class_()

    def serialize(self, query):
        return str(query)

class SortKey:
    """
    Handles comparison operators for sorting

    :param obj: The object to be sorted (a row)
    :param sort_fields: A list of tuples, where each tuple is a pair of (bool, int)
        The bool indicates whether the sort is descending (True) or ascending (False)
        The int indicates the column index to sort on
    """
    def __init__(self, obj, sort_fields, *args):
        self.obj = obj
        self.sort_fields = sort_fields
    def mycmp(self, a, b, sort_fields):
        for desc, colidx in sort_fields:
            if desc:
                if a[colidx] < b[colidx]:
                    return 1
                elif a[colidx] > b[colidx]:
                    return -1
            else:
                if a[colidx] < b[colidx]:
                    return -1
                elif a[colidx] > b[colidx]:
                    return 1
        return 0

    def __lt__(self, other):
        return self.mycmp(self.obj, other.obj, self.sort_fields) < 0

    def __gt__(self, other):
        return self.mycmp(self.obj, other.obj, self.sort_fields) > 0

    def __eq__(self, other):
        return self.mycmp(self.obj, other.obj, self.sort_fields) == 0

    def __le__(self, other):
        return self.mycmp(self.obj, other.obj, self.sort_fields) <= 0

    def __ge__(self, other):
        return self.mycmp(self.obj, other.obj, self.sort_fields) >= 0

    def __ne__(self, other):
        return self.mycmp(self.obj, other.obj, self.sort_fields) != 0

class SortKeyExpressions:
    """
    Handles comparison operators for sorting

    :param obj: The object to be sorted (a row)
    :param sort_expressions: A list of tuples of SqlExpression objects to be used for comparison
        each tuple is a boolean indicating whether the sort is descending (True) or ascending (False)
        followed by the SqlExpression object to be used for comparison.
    :param binding_col_names: A list of column names to be used for binding the sort expression
    """
    def __init__(self, obj, sort_expressions, binding_col_names, *args):
        self.sort_expressions = sort_expressions
        self.bindings = dict((name.lower(), val) for name, val in zip(binding_col_names, obj))
    def mycmp(self, bindings_a, bindings_b, sort_expressions):
        for desc, expr in sort_expressions:
            try:
                v_a = expr.evaluate(bindings_a)
                v_b = expr.evaluate(bindings_b)
                if desc:
                    if v_a < v_b:
                        return 1
                    elif v_a > v_b:
                        return -1
                else:
                    if v_a < v_b:
                        return -1
                    elif v_a > v_b:
                        return 1
            except Exception as e:
                message = f"Error evaluating sort expression {expr}"
                message += "\nWe can only sort using expressions that can be evaluated on output columns."
                raise ValueError(message) from e
        return 0

    def __lt__(self, other):
        return self.mycmp(self.bindings, other.bindings, self.sort_expressions) < 0

    def __gt__(self, other):
        return self.mycmp(self.bindings, other.bindings, self.sort_expressions) > 0

    def __eq__(self, other):
        return self.mycmp(self.bindings, other.bindings, self.sort_expressions) == 0

    def __le__(self, other):
        return self.mycmp(self.bindings, other.bindings, self.sort_expressions) <= 0

    def __ge__(self, other):
        return self.mycmp(self.bindings, other.bindings, self.sort_expressions) >= 0

    def __ne__(self, other):
        return self.mycmp(self.bindings, other.bindings, self.sort_expressions) != 0
