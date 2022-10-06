from enum import Enum

class ColumnType(Enum):
    CATEGORICAL = 1
    ORDINAL = 2
    CONTINUOUS = 3
    
_COL_TYPES = {
    'categorical': ["str", "int"],
    'ordinal': ["int"],
    'continuous': ["float", "int"],
}



