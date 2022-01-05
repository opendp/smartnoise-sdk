##########
SmartNoise
##########

SmartNoise is a set of tools for creating differentially private reports, dashboards, synopses, and synthetic data releases.  It includes a SQL processing layer, supporting queries over Spark and popular database engines, and a collection of synthesizers.

SmartNoise includes a SQL processing library and a synthetic data library.  SmartNoise is built on OpenDP.

|smartnoise-fig-education| |smartnoise-fig-simulations| |smartnoise-fig-size| |smartnoise-fig-utility|

When to Use
===========

Differential privacy is the gold standard definition of privacy.  Use differential privacy when you need to protect your data releases against membership inference, database reconstruction, record linkage, and other privacy attacks.

Here are some rules of thumb for when to use which components:

* Use `OpenDP <http://docs.opendp.org>`_ directly, if you are creating Jupyter notebooks and reproducible research, or if you need the fine-grained control over processing and privacy budget.
* Use `SmartNoise SQL <sql/index.html>`_, if you are generating reports or data cubes over tabular data stored in SQL databases or Spark, or when your data are very large.
* Use `SmartNoise Synthesizers <synth/index.html>`_, if you can't predict the workload in advance, and want to be able to share "looks like" data with collaborators.

Getting Started
===============

.. role:: bash(code)
   :language: bash

For SmartNoise SQL, :bash:`pip install smartnoise-sql` and `read the SQL documentation <sql/index.html>`_

For SmartNoise Synthesizers, :bash:`pip install smartnoise-synth` and `read the Synthesizers documentation <synth/index.html>`_

OpenDP is included with SmartNoise.  To install standalone, :bash:`pip install opendp` and `read the OpenDP documentation <http://docs.opendp.org>`_

Source Code
===========

The SmartNoise GitHub repository is at https://github.com/opendp/smartnoise-sdk

Getting Help
============

If you have questions or feedback regarding SmartNoise, you are welcome to post to the `SmartNoise section`_ of GitHub Discussions.

.. _Smartnoise section: https://github.com/opendp/opendp/discussions/categories/smartnoise




.. |smartnoise-fig-education| image:: _static/images/figs/example_education.png
   :class: img-responsive
   :width: 20%

.. |smartnoise-fig-simulations| image:: _static/images/figs/example_simulations.png
   :class: img-responsive
   :width: 20%

.. |smartnoise-fig-size| image:: _static/images/figs/example_size.png
   :class: img-responsive
   :width: 20%

.. |smartnoise-fig-utility| image:: _static/images/figs/example_utility.png
   :class: img-responsive
   :width: 20%