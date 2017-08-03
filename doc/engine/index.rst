.. FreeDiscovery documentation master file, created by
   sphinx-quickstart on Sun Jul 10 13:58:31 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


FreeDiscovery Engine
====================

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

See http://freediscovery.io/doc/stable/examples/ for more complete examples.

We would very much appreciate feedback on the existing functionality. Feel free to open new issues on Github or send any comments to the mailing list https://groups.google.com/forum/#!forum/freediscovery-ml.

.. toctree::
   :maxdepth: 2


   installation_instructions
   quickstart
   examples/index
   cli
   deployement
   docker
