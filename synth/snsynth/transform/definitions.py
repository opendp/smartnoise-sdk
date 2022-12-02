from enum import Enum

class ColumnType(Enum):
    CATEGORICAL = 1
    ORDINAL = 2
    CONTINUOUS = 3
    UNBOUNDED = 4

