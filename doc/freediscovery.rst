Python API reference
====================

**Note:** the Python API reference will change in future releases (i.e. it is not considered stable).

Two types of classes can be found in FreeDiscovery,
  - scikit-learn compatible estimators that inherit from `sklearn.base.BaseEstimator`
  - freediscovery specific classes that add a persistance layer and are designed to function together with the REST API.

1. Feature extraction
---------------------


.. automodule:: freediscovery.text
    :members:
    :undoc-members:
    :show-inheritance:

2. Document categorization (ML)
-------------------------------

.. automodule:: freediscovery.categorization
    :members:
    :undoc-members:
    :show-inheritance:

3. Document categorization (LSI)
--------------------------------

.. automodule:: freediscovery.lsi
    :members:
    :undoc-members:
    :show-inheritance:

4. Document clustering
-------------------------------

.. automodule:: freediscovery.cluster.base
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: freediscovery.cluster.dendrogram
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: freediscovery.cluster.utils
    :members:
    :undoc-members:
    :show-inheritance:

5. Duplicates detection
-----------------------

.. automodule:: freediscovery.dupdet.base
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: freediscovery.dupdet.simhash
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: freediscovery.dupdet.imatch
    :members:
    :undoc-members:
    :show-inheritance:

5. Tools
--------

.. automodule:: freediscovery.base
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: freediscovery.io
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: freediscovery.utils
    :members:
    :undoc-members:
    :show-inheritance:

6. Datasets
-----------

.. automodule:: freediscovery.datasets
    :members:
    :undoc-members:
    :show-inheritance:

7. Metrics
----------

.. automodule:: freediscovery.metrics
    :members:
    :undoc-members:
    :show-inheritance:

8. Exceptions
-------------
.. automodule:: freediscovery.exceptions
    :members:
    :undoc-members:
    :show-inheritance:
