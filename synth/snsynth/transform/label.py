from snsynth.transform.definitions import ColumnType
from .base import ColumnTransformer
import numpy as np

class LabelTransformer(ColumnTransformer):
    """Transforms categorical values into integer-indexed labels.  Labels will be sorted if possible,
    so that the output can be used as an ordinal.  The indices will be 0-based.

    :param nullable: If null values are expected, a second output will be generated indicating null.
    """
    def __init__(self, nullable=True):
        super().__init__()
        self.nullable = nullable
    @property
    def output_type(self):
        return ColumnType.CATEGORICAL
    @property
    def cardinality(self):
        return [len(self.categories)]
    def _fit(self, val):
        if isinstance(val, float) and np.isnan(val):
            val = None
        if val not in self.labels:
            self.labels[val] = self.category
            self.categories[self.category] = val
            self.category += 1
            self.output_width = 1
    def _fit_finish(self):
        self._fit_complete = True

        # try sorting the categories so this can be used in ordinals
        vals = [v for v in self.labels.keys() if v is not None]
        val_types = set([type(v) for v in vals])
        if len(val_types) > 1:
            return
        sorted_labels = sorted(vals)
        self.labels = {}
        self.categories = []
        for i, label in enumerate(sorted_labels):
            self.labels[label] = i
            self.categories.append(label)
        if self.nullable:
            idx = len(self.categories)
            self.labels[None] = idx
            self.categories.append(None)
        return

    def _clear_fit(self):
        self._reset_fit()
        self.labels = {}
        self.categories = {}
        self.category = 0
    def _transform(self, val):
        if isinstance(val, float) and np.isnan(val):
            val = None
        return self.labels[val]
    def _inverse_transform(self, val):
        if val is None and self.nullable:
            return None
        return self.categories[val]
