# Installation Instructions


## 1. Downloading FreeDiscovery

The latest development version of FreeDiscovery can be obtained from [Github](https://github.com/FreeDiscovery/FreeDiscovery) with,
      
```bash
git clone https://github.com/FreeDiscovery/FreeDiscovery.git
```

It is recommended to use the latest stable version with,
  
```bash
git checkout v1.1.2
```

Althernaitively, the latest stable version can be donwloaded from the [Github releases page](https://github.com/FreeDiscovery/FreeDiscovery/releases).


## 2. Installing the Dependencies

 1. Download and install [Miniconda](http://conda.pydata.org/miniconda.html) 64 bit for Python 3.5 (a cross-platform package manager for Python & R)

 2. A virtual environment with all the dependencies can be setup with the following commands,
 
    ```bash
    cd FreeDiscovery
    conda config --append channels conda-forge
    conda create -n freediscovery-env --file requirements.txt python=3.6

    source activate freediscovery-env   # on Linux/Mac OS X
    # Note: on Windows the above command is "activate freediscovery-env"

    # (optional dependencies, Linux only) 
    conda install simhash-py gunicorn

    pip install -e .
    ```

 3. [optional] The test suite can then be run with,
 
    ```bash
    py.test -sv .
    ```


**Note 1**: all of the above commands (except the installation of the compiler) should be run without `sudo` and with regular user permissions.

**Note 2**: is recommended to use conda in a virtual environment for reproducibility. However, it is also possible to use system Python (3.5 or 3.6), and install the list of dependencies manually (see the quickstart in the README)


      
## 3. Starting the FreeDiscovery Server

The FreeDiscovery server can be started with,

```py   
freediscovery run
```

see CLI documentation for additional details.
