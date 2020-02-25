"""
    Implements engine-specific identifier matching rules
    for escaped identifiers.
"""
class BaseNameCompare:
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
        return value.replace('"','').replace('`','').replace('[','').replace(']','')
    """
        True if any part of identifier is escaped
    """
    def is_escaped(self, identifier):
        return any([p[0] in ['"', '[', '`'] for p in identifier.split('.') if p != ""])
    """
        Converts proprietary escaping to SQL-92.  Supports multi-part identifiers
    """
    def clean_escape(self, identifier):
        escaped = []
        for p in identifier.split('.'):
            if self.is_escaped(p):
                escaped.append(p.replace('[', '"').replace(']', '"').replace('`', '"'))
            else:
                escaped.append(p.lower())
        return '.'.join(escaped)
    """
        Returns true if an identifier should
        be escaped.  Checks only one part per call.
    """
    def should_escape(self, identifier):
        if self.is_escaped(identifier):
            return False
        if identifier.lower() in self.reserved():
            return True
        if identifier.lower().replace(' ', '') == identifier.lower():
            return False
        else:
            return True
        