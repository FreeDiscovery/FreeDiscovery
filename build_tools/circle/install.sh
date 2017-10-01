#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.
#
# This script is adapted from a similar script from the scikit-learn repository.
#
# License: 3-clause BSD

set -x
set -e

create_new_conda_env() {
    # Skip Travis related code on circle ci.
    if [ -z $CIRCLECI ]; then
        # Deactivate the travis-provided virtual environment and setup a
        # conda-based environment instead
        deactivate
    fi

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O ~/miniconda.sh
    chmod +x ~/miniconda.sh && ~/miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH
    echo $PATH
    conda update --quiet --yes conda

    conda config --append channels conda-forge

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --quiet --yes --file requirements_engine.txt simhash-py graphviz python-graphviz python=3.6
    source activate testenv

}


create_new_conda_env

conda install -y pillow nose
pip install -r ./build_tools/requirements_extra_pip.txt
pip install gunicorn


# Build and install scikit-learn in dev mode
python setup.py develop

mkdir -p ../freediscovery_shared
