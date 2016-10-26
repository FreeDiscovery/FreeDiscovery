#!/bin/bash

export FD_CACHE_DIR="$(cd "../$(dirname .)"; pwd)/freediscovery_shared/"
mkdir -p ../freediscovery_shared


echo "
    Download the benchmark TAR dataset (only the first time)
    "
if [ ! -f "${FD_CACHE_DIR}tar_fd_benchmark.tar.gz" ]; then
    curl "http://r0h.eu/d/tar_fd_benchmark.tar.gz" -L -o "${FD_CACHE_DIR}/tar_fd_benchmark.tar.gz"
    cd ${FD_CACHE_DIR} && tar xzf tar_fd_benchmark.tar.gz && cd -
fi

echo "
Starting FREEDiscovery Server $(tail -n 1 freediscovery/_version.py |  cut -d "=" -f2) ($(date))
    shared folder set to ${FD_CACHE_DIR}
 " | tee -a ${FD_CACHE_DIR}/freediscovery-backend.log

source activate freediscovery-env
python scripts/run_api.py ${FD_CACHE_DIR}
