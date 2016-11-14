Python API reference
====================

**Note:** the Python API reference will change in future releases (i.e. it is not considered stable).

Two types of classes can be found in FreeDiscovery,
  - scikit-learn compatible estimators that inherit from `sklearn.base.BaseEstimator`
  - freediscovery specific classes that add a persistance layer and are designed to function together with the REST API.

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

    freediscovery.text.FeatureVectorizer


Document categorization
-----------------------

.. autosummary::
    :toctree: ./generated/

    freediscovery.categorization.Categorizer
    freediscovery.lsi.LSI

Document clustering
--------------------

.. autosummary::
    :toctree: ./generated/

    freediscovery.cluster.Clustering
    freediscovery.cluster.ClusterLabels
    freediscovery.cluster._DendrogramChildren
    freediscovery.cluster.utils._binary_linkage2clusters
    freediscovery.cluster.utils._merge_clusters

Duplicates detection
--------------------

.. autosummary::
    :toctree: ./generated/

    freediscovery.dupdet.DuplicateDetection
    freediscovery.dupdet.SimhashDuplicates
    freediscovery.dupdet.IMatchDuplicates

Tools
-----

.. autosummary::
    :toctree: ./generated/

    freediscovery.base.BaseEstimator
    freediscovery.io.parse_ground_truth_file
    freediscovery.utils.filter_rel_nrel
    freediscovery.utils.generate_uuid
    freediscovery.utils.setup_model

Metrics
-------

This module aims extends `sklearn.metrics <http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_ with a few additional metrics.

.. autosummary::
    :toctree: ./generated/

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
