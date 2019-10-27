"""
    List of SQL reserved keywords that need to be escaped in object names.
    This is used for metadata inference, but may be incorporated in the DataReader
    interface at some point.
"""

sql_reserved = ["group", "select", "on", "where"]