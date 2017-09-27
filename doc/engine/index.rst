FreeDiscovery Engine
====================

**Installation**

FreeDiscovery requires **Python 3.5+** and can be installed with `conda <https://conda.io/>`_,

.. code:: bash

    conda config --append channels conda-forge
    conda install freediscovery


**Running the server**

* ``freediscovery run``
* to check that the server started successfully, ``curl -X GET http://localhost:5001/``



.. toctree::
   :maxdepth: 2

   installation_instructions
   quickstart
   examples/index
   cli
   data_ingestion
   deployment
   API <../openapi-docs/index.html#http://>
