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

    def schema_match(self, query, meta):
        if query.strip() == "" and meta in self.search_path:
            return True
        return self.identifier_match(query, meta)

    """
        Uses database engine matching rules to report True
        if identifier used in query matches identifier
        of metadata object.  Pass in one part at a time.
    """

    def identifier_match(self, query, meta):
        return query == meta

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
