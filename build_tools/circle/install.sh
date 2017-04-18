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

create_new_venv() {
    # At the time of writing numpy 1.9.1 is included in the travis
    # virtualenv but we want to be in control of the numpy version
    # we are using for example through apt-get install
    deactivate
    virtualenv --system-site-packages testvenv
    source testvenv/bin/activate
    pip install nose
}

create_new_conda_env() {
    # Skip Travis related code on circle ci.
    if [ -z $CIRCLECI ]; then
        # Deactivate the travis-provided virtual environment and setup a
        # conda-based environment instead
        deactivate
    fi

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh \
        -O ~/miniconda.sh
    chmod +x ~/miniconda.sh && ~/miniconda.sh -b
    export PATH=$HOME/miniconda2/bin:$PATH
    echo $PATH
    conda update --quiet --yes conda

    conda config --append channels conda-forge

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --quiet --yes --file requirements.txt simhash-py python=3.6
    source activate testenv

}


create_new_conda_env

pip install -r ./build_tools/requirements_extra_pip.txt
pip install gunicorn


# Build and install scikit-learn in dev mode
python setup.py develop

# start the FreeDiscovery server in the background
mkdir -p ../freediscovery_shared
