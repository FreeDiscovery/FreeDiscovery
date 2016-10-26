# Examples

The examples of use for FreeDiscovery can be found in the [`FreeDiscovery/examples`](https://github.com/FreeDiscovery/examples) repository. 

These are Python scripts executed in [Jupyter notebooks](https://jupyter.org/) that illustrate the use of the FreeDiscovery REST API on the 37,000 documents subset of the TREC 2009 legal collection. 

Both the rendered html and the raw Jupyter notebook files (`.ipynb`) are included.

These examples can be run with the following steps,

1. Extract the contents of `FreeDiscovery_examples.zip` under `FreeDiscovery/examples`
2. Use the [direct install](./installation_instructions.html) of FreeDiscovery with conda.
3. The training/test dataset is automatically downloaded when starting FreeDiscovery ([except for Windows](./installation_instructions.html#on-windows))
4. Activate the previously created virtual environment,
 
       source activate freediscovery-env

5. Install jupyter and a few additional dependencies

       pip install -r build_tools/requirements_extra_pip.txt

5. Finally, start jupyter notebook, 

       cd FreeDiscovery/examples
       jupyter notebook
and open the examples.

