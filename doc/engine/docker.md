# Docker setup


## 1. Downloading FreeDiscovery

The latest development version of FreeDiscovery can be obtained from [Github](https://github.com/FreeDiscovery/FreeDiscovery) with,
      
    git clone https://github.com/FreeDiscovery/FreeDiscovery.git

or by downloading the latest stable version from the [Github releases page](https://github.com/FreeDiscovery/FreeDiscovery/releases).


## 2. Building the container

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
   
    bash docker_run.sh freediscovery/freediscovery:<version>

for the docker container, respectively. The repository shared with the Docker container (used to ingest the data) is `../freediscovery_shared`.

### 3.2. On Windows

The server initialization is not currently fully scripted on Windows, and the following actions are necessary,

1. [only once] Create a `FreeDiscovery\..\freediscovery_shared` folder
2. The server can then be started with,

        docker run -t -i -v /<absolute-path-to-the-foder-with-data>:/freediscovery_shared -p 5001:5001 freediscovery/freediscovery:<version>

   for the docker install.
