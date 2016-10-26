# Contributing

This section aims to regroup useful information for contributing to FreeDiscovery. 


## Installation

The direct install using Anaconda Python distribution is recommended for development (particularly on Windows), as described in the corresponding [documentation section](./installation_instructions.html). However, it is also possible to use e.g. system Python, and install the list of dependencies (split between `script/requirements_conda.txt` and `scripts/requirements_pip_*.txt` under `build_tools/`) with `pip`. 

The issue tracker is located at: [https://github.com/dagr1234/FreeDiscoveryBeta](https://github.com/dagr1234/FreeDiscoveryBeta).

The [scikit learn's developper guide](http://scikit-learn.org/stable/developers/index.html) is also worth reading.

## Test suite

A two level test suite is implemented in FreeDiscovery, that validates both the algorithms and the REST API, 

The tests are located under `freediscovery/tests/` and can be run with,

    py.test -s FreeDiscovery/freediscovery/

or alternatively from Python with,

    import freediscovery.tests as ft; ft.run()

It is automatically run as part of the installation procedure locally. The Continuous Integration also runs this test suite on Linux and Windows for all commits and pull requests on GitHub.  

## Building documentation

This html documentation can be built from sources with,

    cd FreeDiscovery/doc/
    make html

which requires to install `sphinx` and `recommonmark`. Alternatively  `make latexpdf` generates documentation in .pdf format (requires `pdflatex`). 


