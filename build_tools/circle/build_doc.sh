# Adapted from scikit-learn
set -x
set -e

# Introspect the commit to know whether or not we should skip building the
# documentation: a pull request that does not change any file in doc/ or
# examples/ folder should be skipped unless the "[doc: build]" is found the
# commit message.
BUILD_DOC=`python build_tools/circle/check_build_doc.py`
echo -e $BUILD_DOC
if [[ $BUILD_DOC == "SKIP:"* ]]; then
    touch ~/log.txt  # the "test" segment needs that file
    exit 0
fi

# Installing required system packages to support the rendering of match
# notation in the HTML documentation
sudo -E apt-get -yq update
sudo -E apt-get -yq remove texlive-binaries --purge
sudo -E apt-get -yq --no-install-suggests --no-install-recommends --force-yes \
    install dvipng texlive-latex-base texlive-latex-extra

sudo -E apt-get -yq install libatlas-dev libatlas3gf-base
sudo -E apt-get -yq install build-essential python-dev python-setuptools wget curl
# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
  deactivate
fi

pushd .
# Install dependencies

pip install --upgrade pip
pip install --upgrade numpy
pip install -r ./build_tools/requirements_conda.txt
pip install -r ./build_tools/requirements_pip_unix.txt
pip install -r ./build_tools/requirements_extra_pip.txt
pip install mock
pip install sphinx-gallery pillow

# Build and install scikit-learn in dev mode
python setup.py develop

# start the FreeDiscovery server in the background
mkdir -p ../freediscovery_shared

if [ ! -f "../freediscovery_shared/treclegal09_2k_subset.tar.gz" ]; then
    curl "http://r0h.eu/d/treclegal09_2k_subset.tar.gz" -L -o "../freediscovery_shared/treclegal09_2k_subset.tar.gz"
    cd ../freediscovery_shared && tar xzf treclegal09_2k_subset.tar.gz && cd -
fi
