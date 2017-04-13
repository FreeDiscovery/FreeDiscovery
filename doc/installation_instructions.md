# Installation Instructions


## 1. Downloading FreeDiscovery

The latest development version of FreeDiscovery can be obtained from [Github](https://github.com/FreeDiscovery/FreeDiscovery) with,
      
    git clone https://github.com/FreeDiscovery/FreeDiscovery.git

or by downloading the latest stable version from the [Github releases page](https://github.com/FreeDiscovery/FreeDiscovery/releases).


## 2. Installing the Dependencies

 1. Download and install [Miniconda](http://conda.pydata.org/miniconda.html) 64 bit for Python 3.5 (a cross-platform package manager for Python & R)

 2. A virtual environment with all the dependencies can be setup with the following commands,
 
          cd FreeDiscovery
          conda config --append channels conda-forge
          conda create -n freediscovery-env --file requirements.txt python=3.6
 
          source activate freediscovery-env   # on Linux/Mac OS X
          # or "activate freediscovery-env"   # on Windows 

          # (optional dependencies, Linux only) 
          conda install -r build_tools/requirements_pip_comp.txt 

          python setup.py develop
 
 3. [optional] The test suite can then be run with,
 
          python -c "import freediscovery.tests as ft; ft.run()"


**Note 1**: all of the above commands (except the installation of the compiler) should be run without `sudo` and with regular user permissions.

**Note 2**: is recommended to use conda in a virtual environment for reproducibility. However, it is also possible to use system Python (3.5 or 3.6), and install the list of dependencies (split between `requirements.txt` and `scripts/requirements_pip*.txt` under `build_tools/`) with `pip`.


      
## 3. Starting the FreeDiscovery Server

### 3.1. On Linux/Mac OS 

The FreeDiscovery server can be started with,
   
    bash conda_run.sh

### 3.2. On Windows

The server initialization is not currently fully scripted on Windows, and the following actions are necessary,

1. [only once] Create a `FreeDiscovery\..\freediscovery_shared` folder
2. The server can then be started with,

        python scripts\run_api.py ..\freediscovery_shared 
