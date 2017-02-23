# Installation Instructions

There are two ways of installing FreeDiscovery,

1. An installation into a Python virtual environment (recommended for development and testing)
2. Using a Docker container (recommended in production)

## 1. Downloading FreeDiscovery

The latest development version of FreeDiscovery can be obtained from [Github](https://github.com/FreeDiscovery/FreeDiscovery) with,
      
    git clone https://github.com/FreeDiscovery/FreeDiscovery.git

or by downloading the latest stable version from the [Github releases page](https://github.com/FreeDiscovery/FreeDiscovery/releases).


## 2. Installing the Dependencies

### 2.a. Python Install

 1. Download and install [Miniconda](http://conda.pydata.org/miniconda.html) 64 bit for Python 3.5 (a cross-platform package manager for Python & R)

 2. To install the optional simhash-py dependency, a g++ (GCC >= 4.8) compiler is required on Linux (on Windows it is not installed). 
 
 3. A virtual environment with all the dependencies can be setup with the following command,
 
          cd FreeDiscovery
          conda create -n freediscovery-env --file requirements.txt python=3.6
 
          source activate freediscovery-env   # on Linux/Mac OS X
          # or "activate freediscovery-env"   # on Windows 

          # (optional, Linux only) requires g++ compiler
          pip install -r build_tools/requirements_pip_comp.txt 

          python setup.py develop
 
 4. [optional] The test suite can then be run with,
 
          python -c "import freediscovery.tests as ft; ft.run()"


**Note**: for convenience, running steps 2 and 3 can be automated:

- on Linux and Mac OS X:

          bash build_tools/conda_setup.sh

- on Windows by double-clicking on `build_tools/conda_setup.bat` in files explorer, or running,

          cd build_tools
          conda_setup.bat

**Note 2**: is recommended to use conda in a virtual environment for reproducibility. However, it is also possible to use system Python (3.5 or 3.6), and install the list of dependencies (split between `requirements.txt` and `scripts/requirements_pip*.txt` under `build_tools/`) with `pip`.


### 2.b. Docker Container
1. Download and [install Docker](https://docs.docker.com/engine/installation/)

2. Download the pre-build container images (requires authentication),

        docker pull "freediscovery/freediscovery:<version>"

   or build the container locally,
   
        cd FreeDiscovery
        docker build -t "freediscovery/freediscovery:<version>" .     

   where `<version>` is the stable version `0.7` or latest development version with the tag `latest`.

      
## 3. Starting the FreeDiscovery Server

### 3.1. On Linux/Mac OS 

The FreeDiscovery server can be started with,
   
    bash conda_run.sh
for the Python install, or with

    bash docker_run.sh freediscovery/freediscovery:<version>
for the docker container, respectively. The repository shared with the Docker container (used to ingest the data) is `../freediscovery_shared`.

### 3.2. On Windows

The server initialization is not currently fully scripted on Windows, and the following actions are necessary,

1. [only once] Create a `FreeDiscovery\..\freediscovery_shared` folder
2. The server can then be started with,

        python scripts\run_api.py ..\freediscovery_shared 
   for the Python install, or with,

        docker run -t -i -v /<absolute-path-to-the-foder-with-data>:/freediscovery_shared -p 5001:5001 freediscovery/freediscovery:<version>

   for the docker install.
