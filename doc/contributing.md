# Contributing

This section aims to regroup useful information for contributing to FreeDiscovery. 


## Submitting bugs

Please feel free to open an issue in the GitHub issue tracker at [https://github.com/FreeDiscovery/FreeDiscovery](https://github.com/FreeDiscovery/FreeDiscovery/issues) for any problem that you may encounter.

Running the test suite (cf. below) may also help in diagnosing the source of the problem.

## Installing the developpement version

To install a the developpement version of FreeDiscovery,

1. If you do not have [Miniconda](https://conda.io/miniconda.html) installed, follow [this tutorial](https://github.com/symerio/tutorials/blob/master/development-practices/install_miniconda.md).

2. If you haven't done it before, add [conda-forge](https://conda-forge.org/) to the conda channels,

   ```bash
   conda config --append channels conda-forge
   ```
3. Download and install the developpement verision of FreeDiscovery,

   ```bash
   git clone https://github.com/FreeDiscovery/FreeDiscovery.git
   cd FreeDiscovery
   conda create -n fd-env --file requirements_engine.txt python=3.6
   source activate fd-env
   pip install -e .
   ```
   this will create a virtualenv called `fd-env` which you can activate with
   `source activate fd-env` then start freediscovery with `freediscovery run`.

   **Note:** if you already have the `fd-env` environement, you may need to delete
   it first with `conda env remove -n fd-env`.

## Contributing

This section describes the workflow for creating Pull Requests (PR). For more complex contributions, it may also be useful to create an associated issue. 

 1. Fork the FreeDiscovery repository
 2. Clone this fork on your computer and [install it](./engine/installation_instructions.html)
 3. Make the appropriate changes
 4. Make sure that the tests suite (cf. below) does not produce errors
 5. Commit and push the changes to GitHub
 6. Create a Pull Request from your branch to the `master` branch
 7. If any of the continuous integration services (Travis CI, Appveyor CI, Circle CI) produce an error, review the corresponding output and fix the code if appropriate. 
 8. After the PR is merged, this branch can be safely deleted from your fork (and a new one may be created for subsequent contributions).



## Test Suite

A two level test suite is implemented in FreeDiscovery, that validates both the algorithms and the REST API, 

The tests are located under `freediscovery/tests/` and can be run with,

```bash
py.test -s .
```


## Building Documentation

### 1. Sphinx documentation 

The html documentation can be built from sources with,
     
```bash
# starting the FreeDiscovery server at localhost
cd FreeDiscovery/doc/
make html
```

which requires to install dependencies in `build_tools/requirements_extra_pip.txt`. This would also run and include examples using `sphinx-gallery`.

Alternatively  `make latexpdf` generates documentation in .pdf format (requires `pdflatex`). 

### 2. REST API documentation

The rest API documentation can be generated with,

```bash
sudo npm install -g bootprint 
sudo npm install -g bootprint-openapi
bootprint openapi http://0.0.0.0:5001/openapi-specs.json openapi_docs

cp -r openapi_docs/ _build/html/
```
