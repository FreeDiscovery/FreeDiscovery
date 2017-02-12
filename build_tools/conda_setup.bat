REM to run from Windows command line: cd into build_tools folder and type conda_setup.bat
REM to run from files explorer: double-click conda_setup.bat

cd ..
call deactivate
conda env remove --yes -n freediscovery-env
conda create -n freediscovery-env --yes --file requirements.txt python=3.6
call activate freediscovery-env
python setup.py develop
python -c "import freediscovery.tests as ft; ft.run()"
