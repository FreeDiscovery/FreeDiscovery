FreeDiscovery Engine
====================


FreeDiscovery requires **Python 3.5+** and can be installed with 

.. code:: bash

    pip install freediscovery[engine]


or alternatively with `conda <https://conda.io/>`_,

.. code:: bash

    conda install -c conda-forge freediscovery


The server can be started with

.. code:: bash

    freediscovery run

to check that the server is successfully runnining, open ``http://localhost:5001/``.


Contents
^^^^^^^^

.. toctree::
   :maxdepth: 2

   overview
   quickstart
   examples/index
   data_ingestion
   cli
   deployment
   API <../openapi-docs/index.html#http://>
