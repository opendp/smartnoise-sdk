==========================
Synthesizers API Reference
==========================

MWEM
----

.. autoclass:: snsynth.MWEMSynthesizer
    :members:
    :undoc-members:
    :show-inheritance:

QUAIL
-----

.. autoclass:: snsynth.QUAILSynthesizer
    :members:
    :undoc-members:
    :show-inheritance:

PyTorch
-------

.. autoclass:: snsynth.pytorch.PytorchDPSynthesizer
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: snsynth.pytorch.nn.PATEGAN
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: snsynth.pytorch.nn.DPGAN
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: snsynth.pytorch.nn.PATECTGAN
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

============
Transformers
============

Column Transformers
-------------------

.. autoclass:: snsynth.transform.minmax.MinMaxTransformer
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: snsynth.transform.label.LabelTransformer
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: snsynth.transform.onehot.OneHotEncoder
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

