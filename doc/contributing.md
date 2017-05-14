# Contributing

This section aims to regroup useful information for contributing to FreeDiscovery. 


## Submitting bugs

Please feel free to open an issue in the GitHub issue tracker at [https://github.com/FreeDiscovery/FreeDiscovery](https://github.com/FreeDiscovery/FreeDiscovery/issues) for any problem that you may encounter.

Running the test suite (cf. below) may also help in diagnosing the source of the problem.

## Contributing

This section describes the workflow for creating Pull Requests (PR). For more complex contributions, it may also be useful to create an associated issue. 

 1. Fork the main FreeDiscovery repository
 2. Clone this fork on your computer and [install it](https://freediscovery.github.io/doc/dev/installation_instructions.html#a-python-install)
 3. Make the appropriate changes
 4. Make sure that the tests suite (cf. below) does not produce errors
 5. Commit and push the changes to GitHub
 6. Create a Pull Request from your branch to the `master` branch of the FreeDiscovery repository so it can be reviewed and merged. 
 7. If any of the continuous integration services (Travis CI, Appveyor CI, Circle CI) produce an error, review the corresponding output and fix the code if appropriate. 
 8. After the PR is merged, this branch can be safely deleted from your fork (and a new one may be created for subsequent contributions).



## Test Suite

A two level test suite is implemented in FreeDiscovery, that validates both the algorithms and the REST API, 

The tests are located under `freediscovery/tests/` and can be run with,

    py.test -s FreeDiscovery/freediscovery/

or alternatively from Python with,

    import freediscovery.tests as ft; ft.run()

It is automatically run as part of the installation procedure locally. The Continuous Integration also runs this test suite on Linux and Windows for all commits and pull requests on GitHub.  


## Building Documentation

### 1. Sphinx documentation 

The html documentation can be built from sources with,
     
    # starting the FreeDiscovery server at localhost
    cd FreeDiscovery/doc/
    make html

which requires to install dependencies in `build_tools/requirements_extra_pip.txt`. This would also run and include examples using `sphinx-gallery`.

Alternatively  `make latexpdf` generates documentation in .pdf format (requires `pdflatex`). 

### 2. REST API documentation

The rest API documentation can be generated with,

    sudo npm install -g bootprint 
    sudo npm install -g bootprint-openapi
    bootprint openapi http://0.0.0.0:5001/openapi-specs.json openapi_docs

    cp -r openapi_docs/ _build/html/
