from snsynth.transform.definitions import ColumnType
from .base import ColumnTransformer
import numpy as np

class LabelTransformer(ColumnTransformer):
    """Transforms categorical values into integer-indexed labels.  Labels will be sorted if possible,
    so that the output can be used as an ordinal.  The indices will be 0-based.

    By default the list of categories is learned from the data.  That list is not differentially
    private: every distinct value in the column, including a value held by a single row, becomes a
    category, and no privacy budget is spent on it.  If the set of possible values is public, pass it
    as ``categories`` and the data will not be consulted when fitting.

    :param nullable: If null values are expected, a second output will be generated indicating null.
    :param categories: An optional list of the public categories of this column, in the order to be
        used for the labels.  When supplied, fitting ignores the data.  A value that is not in the list
        is transformed to null if ``nullable`` is True, and raises a ``ValueError`` otherwise.
    """
    def __init__(self, nullable=True, categories=None):
        self.public_categories = None
        if categories is not None:
            categories = list(categories)
            if any(c is None or (isinstance(c, float) and np.isnan(c)) for c in categories):
                raise ValueError("categories must not contain null values; use nullable=True instead")
            if len(set(categories)) != len(categories):
                raise ValueError("categories must not contain duplicates")
            if len(categories) == 0:
                raise ValueError("categories must not be empty")
            self.public_categories = categories
        super().__init__()
        self.nullable = nullable
    @property
    def output_type(self):
        return ColumnType.CATEGORICAL
    @property
    def cardinality(self):
        return [len(self.categories)]
    def _fit(self, val):
        if self.public_categories is not None:
            return  # the category list is public: the data is not consulted
        if isinstance(val, float) and np.isnan(val):
            val = None
        if val not in self.labels:
            self.labels[val] = self.category
            self.categories[self.category] = val
            self.category += 1
            self.output_width = 1
    def _fit_finish(self):
        self._fit_complete = True

        if self.public_categories is not None:
            # keep the caller's order, so that it can be used as an ordinal
            self.labels = {}
            self.categories = []
            for i, label in enumerate(self.public_categories):
                self.labels[label] = i
                self.categories.append(label)
            if self.nullable:
                self.labels[None] = len(self.categories)
                self.categories.append(None)
            self.output_width = 1
            return

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
        if self.public_categories is not None and val not in self.labels:
            if self.nullable:
                return self.labels[None]
            raise ValueError(
                f"Value {val!r} is not in the public categories of this column. "
                "Add it to categories, or use nullable=True to map unknown values to null."
            )
        return self.labels[val]
    def _inverse_transform(self, val):
        if val is None and self.nullable:
            return None
        return self.categories[val]
