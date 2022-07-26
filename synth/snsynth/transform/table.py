

class TableTransformer:
    def __init__(self, transformers):
        # one transformer per column
        self.transformers = transformers
        if self.fit_complete:
            self.output_width = sum([t.output_width for t in self.transformers])
        else:
            self.output_width = 0
    @property
    def fit_complete(self):
        return all([t.fit_complete for t in self.transformers])
    def fit(self, data):
        for t in self.transformers:
            t._clear_fit()
        for row in data:
            self._fit(row)
        for t in self.transformers:
            t._fit_finish()
        self._fit_finish()
    def _fit(self, row):
        for v, t in zip(row, self.transformers):
            t._fit(v)
    def _fit_finish(self):
        self.output_width = sum([t.output_width for t in self.transformers])
    def transform(self, data):
        return [self._transform(row) for row in data]
    def _transform(self, row):
        out_row = []
        for v, t in zip(row, self.transformers):
            if t.output_width == 1:
                out_row.append(t._transform(v))
            else:
                for out_v in t._transform(v):
                    out_row.append(out_v)
        return tuple(out_row)
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    def inverse_transform(self, data):
        return [self._inverse_transform(row) for row in data]
    def _inverse_transform(self, row):
        if len(row) != self.output_width:
            raise ValueError(f"Row has wrong length: got {len(row)}, expected {self.output_width}")
        out_row = []
        row = list(row)
        for t in self.transformers:
            if t.output_width == 1:
                v = row.pop(0)
            else:
                v = tuple([row.pop(0) for _ in range(t.output_width)])
            out_row.append(t._inverse_transform(v))
        return tuple(out_row)
