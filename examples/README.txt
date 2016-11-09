Examples
--------

These Python scripts illustrate the use of the FreeDiscovery REST and Python
APIs on a subset of the TREC 2009 legal collection. The online documentatior
includes results with a 2 000 document subset, but these examples can be
updated to use the 20 000 or 37 000 document subsets as well as the full 700 0000 documents collections.

These examples can be run with the following steps,

 1. Use the `direct install <https://freediscovery.github.io/doc/dev/installation_instructions.html>`_ of FreeDiscovery with conda.
 2. The training/test dataset is automatically downloaded when starting FreeDiscovery (`except for Windows <https://freediscovery.github.io/doc/dev/installation_instructions.html#on-windows>`_)
 3. Activate the previously created virtual environment,

    .. code-block:: bash
 
       source activate freediscovery-env

 4. Install a few additional dependencies

    .. code-block:: bash

       pip install -r build_tools/requirements_extra_pip.txt

 6. Finally, run examples, e.g. 
     
    .. code-block:: bash

       python examples/categorization_examples.py


