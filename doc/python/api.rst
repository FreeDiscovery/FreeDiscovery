API
===

This is the API reference for the FreeDiscovery Python package

.. currentmodule:: freediscovery

Datasets
--------

.. autosummary::
    :toctree: ./generated/

    freediscovery.datasets.load_dataset


Feature extraction
------------------

.. autosummary::
    :toctree: ./generated/
    :template: autosummary/base.rst
    
    freediscovery.feature_weighting.SmartTfidfTransformer


Document categorization
-----------------------
.. autosummary::
    :toctree: ./generated/
    :template: autosummary/base.rst

    freediscovery.neighbors.NearestNeighborRanker

Document clustering
--------------------

.. autosummary::
    :toctree: ./generated/
    :template: autosummary/base.rst

    freediscovery.cluster.ClusterLabels

Duplicates detection
--------------------

.. autosummary::
    :toctree: ./generated/
    :template: autosummary/base.rst

    freediscovery.near_duplicates.SimhashNearDuplicates
    freediscovery.near_duplicates.IMatchNearDuplicates

    
Semantic search
---------------
.. autosummary::
    :toctree: ./generated/
    :template: autosummary/base.rst

    freediscovery.search.Search


IO
--

.. autosummary::
    :toctree: ./generated/
    :template: autosummary/base.rst

    freediscovery.io.parse_smart_tokens

Metrics
-------

This module aims to extend `sklearn.metrics <http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_ with a few additional metrics,

.. autosummary::
    :toctree: ./generated/

    freediscovery.metrics.recall_at_k_score
