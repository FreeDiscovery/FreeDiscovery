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

Data ingestion
~~~~~~~~~~~~~~

1. Create a new vectorized dataset with :bash:``curl -X POST 'http://localhost:5001/api/v0/feature-extraction'`` and save the returned ``dataset_id``.
2. Ingest the dataset,

   .. code:: bash

        curl -X POST 'http://localhost:5001/api/v0/feature-extraction/{dataset_id}' \
             -H 'Content-Type: application/json' -d '
        {
           "data_dir": "./20_newgroups/",
           "document_id_generation": "infer_file_path",
        }'
3. Get the mapping between ``file_path`` of individial files and their ``document_id``:
   
   ``curl -X POST 'http://localhost:5001/api/v0/feature-extraction/{dataset_id}/id-mapping'``

See http://freediscovery.io/doc/stable/examples/ for additional examples.


We would very much appreciate feedback on the existing functionality. Feel free to open new issues on Github or send any comments to the mailing list https://groups.google.com/forum/#!forum/freediscovery-ml.

Documentation
-------------

For more information see the documentation and API Reference,

- development version [``master`` branch | documentation http://freediscovery.io/doc/dev/ ].
- stable version 1.1.1 [``1.1.X`` branch | documentation http://freediscovery.io/doc/stable/ ].

Licence
-------

FreeDiscovery is released under the 3-clause BSD licence.

.. image:: https://freediscovery.github.io/static/grossmanlabs-old-logo-small.gif
    :target: http://www.grossmanlabs.com/
