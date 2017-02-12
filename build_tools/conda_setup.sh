#!/bin/bash

conda env remove --yes -n freediscovery-env
conda config --append channels conda-forge
conda create -n freediscovery-env --yes --file requirements.txt python=3.6

source activate freediscovery-env
python setup.py develop
python -c "import freediscovery.tests as ft; ft.run()"
