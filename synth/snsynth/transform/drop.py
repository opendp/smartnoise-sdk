from .base import ColumnTransformer
from .definitions import ColumnType


class DropTransformer(ColumnTransformer):
    """
    Transformer that ignores a column completely. All values will be dropped during transformation.
    Inverse transformation is a no-op.
    """
    def __init__(self):
        super().__init__()

    @property
    def output_type(self):
        return ColumnType.UNBOUNDED

    @property
    def cardinality(self):
        return [None]

    def _fit(self, _):
        pass

    def _clear_fit(self):
        self._fit_complete = True
        self.output_width = 0

    def _transform(self, _):
        pass

    def _inverse_transform(self, _):
        pass

    def transform(self, data, idx=None):
        if idx is None:
            return [None for _ in data]
        else:
            return [row[:idx] + row[idx + 1 :] for row in data]

    def inverse_transform(self, data, idx=None):
        return data
