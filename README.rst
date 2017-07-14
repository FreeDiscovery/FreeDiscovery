FreeDiscovery
=============

.. image:: https://img.shields.io/pypi/v/freediscovery.svg
    :target: https://pypi.python.org/pypi/freediscovery

.. image:: https://travis-ci.org/FreeDiscovery/FreeDiscovery.svg?branch=master
    :target: https://travis-ci.org/FreeDiscovery/FreeDiscovery

.. image:: https://ci.appveyor.com/api/projects/status/w5kjscmqlrlehp5t/branch/master?svg=true
    :target: https://ci.appveyor.com/project/FreeDiscovery/freediscovery/branch/master

.. image:: https://codecov.io/gh/FreeDiscovery/FreeDiscovery/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/FreeDiscovery/FreeDiscovery


Open Source e-Discovery and Information Retrieval Engine

FreeDiscovery is built on top of existing machine learning libraries (scikit-learn) and provides a REST API for information retrieval applications. It aims to benefit existing e-Discovery and information retrieval platforms with a focus on text categorization, semantic search, document clustering, duplicates detection and e-mail threading.

In addition, FreeDiscovery can be used as Python package and exposes several estimators with a scikit-learn compatible API. 


Installation
------------

FreeDiscovery requires **Python 3.5+** and can be installed with `conda <https://conda.io/>`_: ``conda install -c conda-forge freediscovery``

Alternatively, to install with pip,

1. Install scipy and numpy
2. Run ``pip install freediscovery[all]``


Running the server
------------------

* ``freediscovery run``
* to check that the server started successfully, ``curl -X GET http://localhost:5001/``

Quick start
-----------

1. Install FreeDiscovery and start the server (see above)
2. Download the 20_newsgroup dataset: ``freediscovery download 20_newsgroups``

1. Data ingestion
~~~~~~~~~~~~~~~~~

1. Create a new vectorized dataset with ``curl -X POST http://localhost:5001/api/v0/feature-extraction`` and save the returned hexadecimal ``id`` for later use with ``export FD_DATASET_ID=<returned-id>``.
2. Ingest the dataset,

   .. code:: bash

        curl -X POST -H 'Content-Type: application/json' -d '{
           "data_dir": "./20_newsgroups/"
        }'  http://localhost:5001/api/v0/feature-extraction/${FD_DATASET_ID}
3. Get the mapping between ``file_path`` of individial files and their ``document_id``:
   
   ``curl -X POST http://localhost:5001/api/v0/feature-extraction/${FD_DATASET_ID}/id-mapping > ./fd_id_mapping.txt``
  
   and save the results.

2. Latent Semantic Indexing (LSI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The creation of an LSI index is necessary for clustering, nearest neighbor classification, semantic search and near-duplicates detection,

.. code:: bash

    curl -X POST -H 'Content-Type: application/json' -d "{
       \"parent_id\": \"${FD_DATASET_ID}\"
    }"  http://localhost:5001/api/v0/lsi/

Save the returned ``id`` for later use with ``export FD_LSI_ID=<returned-id>``.



3. Semantic search
~~~~~~~~~~~~~~~~~~

Search in the semantic space can be performed with,

.. code:: bash

    curl -X POST -H 'Content-Type: application/json' -d "{
       \"parent_id\": \"${FD_LSI_ID}\",
       \"query\": \"Jupyter moon\", \"max_results\": 10
     }"  http://localhost:5001/api/v0/search/

4. Categorization
~~~~~~~~~~~~~~~~~

Create a categorization model,

.. code:: bash

    curl -X POST -H 'Content-Type: application/json' -d "{
       \"parent_id\": \"${FD_DATASET_ID}\",
       \"method\": \"LogisticRegression\",
       \"data\": [{\"document_id\": 14000, \"category\": \"sci.space\"},
                  {\"document_id\": 14003, \"category\": \"sci.space\"},
                  {\"document_id\": 18780, \"category\": \"talk.politics.misc\"},
                  {\"document_id\": 18784, \"category\": \"talk.politics.misc\"}
                  ],
       \"training_scores\": true
     }"  http://localhost:5001/api/v0/categorization/

Save the returned ``id`` for later use with ``export FD_CAT_ID=<returned-id>``.

Predictions for the other documents in the dataset can then be retrieved with,

.. code:: bash

    curl -X GET -H 'Content-Type: application/json' -d "{
       \"max_results\": 10, \"max_result_categories\": 2, \"sort_by\": \"sci.space\"
     }"  http://localhost:5001/api/v0/categorization/${FD_CAT_ID}/predict

The correspondence of these results with ground truth categories can be checked in ``fd_id_mapping.txt``.

5. Hierarchical clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a Birch hierarchical clustering model,

.. code:: bash

    curl -X POST -H 'Content-Type: application/json' -d "{
       \"parent_id\": \"${FD_LSI_ID}\",
       \"min_similarity\": 0.7, \"max_tree_depth\": 2
     }"  http://localhost:5001/api/v0/clustering/birch/

Save the returned ``id`` for later use with ``export FD_BIRCH_ID=<returned-id>``.


Finally retrieve the computed hierarchical clusters,

.. code:: bash

    curl -X GET http://localhost:5001/api/v0/clustering/birch/${FD_BIRCH_ID}


See http://freediscovery.io/doc/stable/examples/ for more complete examples.

We would very much appreciate feedback on the existing functionality. Feel free to open new issues on Github or send any comments to the mailing list https://groups.google.com/forum/#!forum/freediscovery-ml.

Documentation
-------------

For more information see the documentation and API Reference,

- development version [``master`` branch | documentation http://freediscovery.io/doc/dev/ ].
- stable version 1.2 [``1.2.X`` branch | documentation http://freediscovery.io/doc/stable/ ].

Licence
-------

FreeDiscovery is released under the 3-clause BSD licence.

|logo1|    |logo2|

.. |logo1| image:: https://freediscovery.github.io/static/freediscovery-full-logo-v2.png
    :scale: 80 %

.. |logo2| image:: https://freediscovery.github.io/static/grossmanlabs-old-logo-small.gif
    :target: http://www.grossmanlabs.com/
