from copy import deepcopy

from snsynth.sdv.rdt.transformers.categorical import LabelEncodingTransformer, OneHotEncodingTransformer
from snsynth.sdv.rdt.transformers.null import NullTransformer
from snsynth.sdv.rdt.transformers.numerical import NumericalTransformer
from snsynth.sdv.rdt.transformers.base import BaseTransformer

__all__ = ["LabelEncodingTransformer", 
           "OneHotEncodingTransformer", 
           "NullTransformer",
           "NumericalTransformer"]

TRANSFORMERS = {
    transformer.__name__: transformer
    for transformer in BaseTransformer.get_subclasses()
}

def get_transformer_instance(transformer):
    """Load a new instance of a ``Transformer``.
    The ``transformer`` is expected to be a ``string`` containing  the transformer ``class``
    name, a transformer instance or a transformer type.
    Args:
        transformer (dict or BaseTransformer):
            ``dict`` with the transformer specification or instance of a BaseTransformer
            subclass.
    Returns:
        BaseTransformer:
            BaseTransformer subclass instance.
    """
    if isinstance(transformer, BaseTransformer):
        return deepcopy(transformer)

    if isinstance(transformer, str):
        transformer = TRANSFORMERS[transformer]

    return transformer()