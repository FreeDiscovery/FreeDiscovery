Python API Reference
====================

Two types of classes can be found in FreeDiscovery,
  - scikit-learn compatible estimators that inherit from `sklearn.base.BaseEstimator`
  - freediscovery specific classes that add a persistance layer and are designed to function together with the REST API.

.. currentmodule:: freediscovery

Datasets
--------

.. autosummary::
    :toctree: ./generated/

    freediscovery.datasets.load_dataset
    freediscovery.ingestion.DocumentIndex

Feature extraction
------------------


.. autosummary::
    :toctree: ./generated/

    freediscovery.text.FeatureVectorizer


Document categorization
-----------------------

.. autosummary::
    :toctree: ./generated/

    freediscovery.categorization._CategorizerWrapper
    freediscovery.categorization.NearestNeighborRanker
    freediscovery.lsi._LSIWrapper

Document clustering
--------------------

.. autosummary::
    :toctree: ./generated/

    freediscovery.cluster._ClusteringWrapper
    freediscovery.cluster.ClusterLabels
    freediscovery.cluster._DendrogramChildren
    freediscovery.cluster.utils._binary_linkage2clusters
    freediscovery.cluster.utils._merge_clusters

Duplicates detection
--------------------

.. autosummary::
    :toctree: ./generated/

    freediscovery.dupdet._DuplicateDetectionWrapper
    freediscovery.dupdet.SimhashDuplicates
    freediscovery.dupdet.IMatchDuplicates

Email threading
---------------

.. autosummary::
    :toctree: ./generated/

    freediscovery.email_threading._EmailThreadingWrapper
    
Semantic search
---------------
.. autosummary::
    :toctree: ./generated/

    freediscovery.search.Search
    freediscovery.search._SearchWrapper


Tools
-----

.. autosummary::
    :toctree: ./generated/

    freediscovery.base._BaseWrapper
    freediscovery.pipeline.PipelineFinder
    freediscovery.utils.generate_uuid
    freediscovery.utils.setup_model


IO
--

.. autosummary::
    :toctree: ./generated/

    freediscovery.io.parse_ground_truth_file
    freediscovery.io.parse_rcv1_smart_tokens

Metrics
-------

This module aims to extend `sklearn.metrics <http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_ with a few additional metrics,

.. autosummary::
    :toctree: ./generated/

    freediscovery.metrics.recall_at_k_score
    freediscovery.metrics.ratio_duplicates_score
    freediscovery.metrics.f1_same_duplicates_score

Exceptions
----------

.. autosummary::
    :toctree: ./generated/

    freediscovery.exceptions.NotFound
    freediscovery.exceptions.DatasetNotFound
    freediscovery.exceptions.ModelNotFound
    freediscovery.exceptions.InitException
    freediscovery.exceptions.WrongParameter
    freediscovery.exceptions.NotImplementedFD
    freediscovery.exceptions.OptionalDependencyMissing
