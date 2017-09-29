Data ingestion
==============


Format of processed files
^^^^^^^^^^^^^^^^^^^^^^^^^

FreeDiscovery can process document collections, where files are stored within a folder hierarchy, with the assumption that each separate file corresponds to a document.

For ingesting a custom dataset, it is necessary is to place the data collection on the FreeDiscovery server (or within the folder mounted inside the Docker container) and provide either the name of the folder where the data is located (``data_dir``) or individual paths to every file in the collection (``file_path``).

.. note::
   
   the `data_dir` and `file_path` should be absolute paths.


Document indexing 
^^^^^^^^^^^^^^^^^

Each document in the collection is given a unique numerical ``internal_id`` that is used to identify documents internally. This field is however not exposed by the REST API. Instead, the following fields can also be used for indexing,

 * ``file_path``: the absolute path to the file. Note that when FreeDiscovery is provided with a list of ``file_path``, the ``data_dir`` will be recomputed as the longest common path.
 * ``document_id``: an external numeric document id provided in the ``POST /api/v0/feature-extraction/<dataset-id>`` step
 * ``document_id`` together with a ``rendition_id``, where the latter is also provided in the ``POST /api/v0/feature-extraction/<dataset-id>`` processing step

To use a field (or a group of fields) as a index, it must be unique, meaning that duplicates are not supported. The only field that is unique by construction is ``internal_id``, all the rest being user provided. Additional information regarding data ingestion and indexing can be found in the `REST_data_ingestion <../examples/REST_data_ingestion.html>`_ example. The mapping between different index fields is provided by

 * ``POST /api/v0/feature-extraction/<id>/id-mapping/nested``


The training of the categorization model also uses a similar mechanism to identify training set documents.

.. note::

   the different fields above can always be associated to the processed documents, however attempting to index (e.g. select) a subset of documents using a field (or a group of fields) with duplicates will result in an error.

Example datasets
^^^^^^^^^^^^^^^^

A `few example datasets <../rest_api/dataset_get.html>`_ (subsets of the TREC Legal 2009 collection) can be automatically downloaded by FreeDiscovery with the following command,

.. code::

   GET /api/v0/dataset/<dataset-name>

These datasets are in particular used in `the examples <../examples/index.html>`_ and contain the ground truth classification labels.
