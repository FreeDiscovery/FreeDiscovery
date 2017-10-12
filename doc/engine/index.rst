FreeDiscovery Engine
====================

FreeDiscovery Engine provides a REST API for information retrieval applications. It aims to benefit existing e-Discovery and information retrieval platforms with a focus on text categorization, semantic search, document clustering, duplicates detection and e-mail threading.

The installation requires **Python 3.5+** and can be done with,

.. code:: bash

    pip install freediscovery[engine]


or alternatively with `conda <https://conda.io/>`_,

.. code:: bash

    conda config --append channels conda-forge
    conda install freediscovery


The server can be started with

.. code:: bash

    freediscovery run

To check that the server is successfully runnining, open ``http://localhost:5001/``.


.. toctree::
   :maxdepth: 1

   overview
   quickstart
   examples/index
   data_ingestion
   cli
   deployment
   API <../openapi-docs/index.html#http://>
