#!/bin/bash

export FD_CACHE_DIR="$(cd "../$(dirname .)"; pwd)/freediscovery_shared/"
mkdir -p ../freediscovery_shared

echo "
Starting FREEDiscovery Server $(tail -n 1 freediscovery/_version.py |  cut -d "=" -f2) ($(date))
    shared folder set to ${FD_CACHE_DIR}
 " | tee -a ${FD_CACHE_DIR}/freediscovery-backend.log

source activate freediscovery-env
python scripts/run_api.py ${FD_CACHE_DIR}
