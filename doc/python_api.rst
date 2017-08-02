Python API Reference
====================

.. currentmodule:: freediscovery

Datasets
--------

.. autosummary::
    :toctree: ./generated/

    freediscovery.datasets.load_dataset
    freediscovery.ingestion.DocumentIndex

Document categorization
-----------------------

.. autosummary::
    :toctree: ./generated/

    freediscovery.categorization.NearestNeighborRanker

Document clustering
--------------------

.. autosummary::
    :toctree: ./generated/

    freediscovery.cluster.ClusterLabels

Duplicates detection
--------------------

.. autosummary::
    :toctree: ./generated/

    freediscovery.dupdet.SimhashDuplicates
    freediscovery.dupdet.IMatchDuplicates

    
Semantic search
---------------
.. autosummary::
    :toctree: ./generated/

    freediscovery.search.Search


IO
--

.. autosummary::
    :toctree: ./generated/

    freediscovery.io_smart_tokens

Metrics
-------

This module aims to extend `sklearn.metrics <http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_ with a few additional metrics,

.. autosummary::
    :toctree: ./generated/

    freediscovery.metrics.recall_at_k_score
    freediscovery.metrics.ratio_duplicates_score
    freediscovery.metrics.f1_same_duplicates_score
