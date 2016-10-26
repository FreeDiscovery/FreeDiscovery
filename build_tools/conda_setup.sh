#!/bin/bash

conda env remove --yes -n freediscovery-env
conda create -n freediscovery-env --yes --file ./build_tools/requirements_conda.txt python=3.5

source activate freediscovery-env
pip install -r ./build_tools/requirements_pip_unix.txt
python setup.py develop 
python -c "import freediscovery.tests as ft; ft.run()"
