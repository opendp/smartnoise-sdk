from .rowset import TypedRowset


class Reader:
    ENGINE = None

    def __init__(self, name_compare, serializer=None):
        self.compare = name_compare
        self.serializer = serializer

    @property
    def engine(self):
        return self.ENGINE

    def execute(self, query):
        raise NotImplementedError("Execute must be implemented on the inherited class")

    def execute_typed(self, query):
        if not isinstance(query, str):
            raise ValueError("Please pass a string to this function.  You can use execute_ast to execute ASTs")

        rows = self.execute(query)
        if len(rows) < 1:
            return None
        types = ["unknown" for i in range(len(rows[0]))]
        if len(rows) > 1:
            row = rows[1]
            for idx in range(len(row)):
                val = row[idx]
                if isinstance(val, int):
                    types[idx] = "int"
                elif isinstance(val, float):
                    types[idx] = "float"
                elif isinstance(val, bool):
                    types[idx] = "boolean"
                else:
                    types[idx] = "string"

        return TypedRowset(rows, types)

    def execute_ast(self, query):
        if isinstance(query, str):
            raise ValueError("Please pass ASTs to execute_ast.  To execute strings, use execute.")
        if hasattr(self, 'serializer') and self.serializer is not None:
            query_string = self.serializer.serialize(query)
        else:
            query_string = str(query)
        return self.execute(query_string)

    def execute_ast_typed(self, query):
        syms = query.all_symbols()
        types = [s[1].type() for s in syms]

        rows = self.execute_ast(query)
        return TypedRowset(rows, types)
