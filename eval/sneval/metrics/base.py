import importlib
import json

class Metric:
    @classmethod
    def create(cls, name, *args, **kwargs):
        position_args = ['column_name', 'column_names', 'categorical_columns', 'measure_columns', 'sum_columns', 'label_column', 'prediction_column']
        if not args:
            args = [kwargs.pop(arg) for arg in position_args if arg in kwargs]
        module = importlib.import_module('sneval.metrics')
        if not hasattr(module, name):
            raise ValueError("Module {} does not have attribute {}.".format(module, name))
        cls = getattr(module, name)
        if not issubclass(cls, Metric):
            raise ValueError("Class {} is not a subclass of Metric.".format(name))
        return cls(*args, **kwargs)
    @property
    def name(self):
        return self.__class__.__name__

    def param_names(self):
        return []
    
    def __eq__(self, other):
        if not isinstance(other, Metric):
            return False
        return self.to_dict() == other.to_dict()
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash(self.serialize())
    
    def __repr__(self):
        return self.serialize()
    
    def to_dict(self):
        out = {
            "name": self.name,
            "parameters": {}
        }
        for param in self.param_names():
            out["parameters"][param] = getattr(self, param)
        return out
    @classmethod
    def from_dict(cls, d):
        if 'name' not in d:
            raise ValueError("Metric dict must have 'name' key.")
        args = []
        kwargs = {}
        for param in d.keys():
            if param not in ['name', 'column', 'columns']:
                kwargs[param] = d[param]
        return cls.create(d['name'], *args, **kwargs)
    def serialize(self):
        return json.dumps(self.to_dict())
    @classmethod
    def deserialize(cls, s):
        return cls.from_dict(json.loads(s))