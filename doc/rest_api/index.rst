REST API Reference
==================

This REST API allows to use FreeDiscovery from any supported programming language. 

Following resources are defined,

Load Benchmark Dataset
----------------------

=======================================================  ======  =========================================================
`/api/v0/dataset/<dataset-name> <./dataset_get.html>`_   GET     Load benchmark dataset
=======================================================  ======  =========================================================

Parsers
-------

======================================================================================  ======  ==========================================================
`/api/v0/email-parser/ <./email_parser_post.html>`_                                     POST    Load a dataset and parse emails
`/api/v0/email-parser/ <./email_parser_get.html>`_                                      GET     List processed datasets
`/api/v0/email-parser/<dataset-id> <./email_parser_element_get.html>`_                  GET     Load parsed emails
`/api/v0/email-parser/<dataset-id>/index <./email_parser_index_get.html>`_              GET     Query document index for a list of filenames
`/api/v0/email-parser/<dataset-id> <./email_parser_delete.html>`_                       DELETE  Delete a processed dataset
======================================================================================  ======  ==========================================================

Feature Extraction 
------------------

======================================================================================  ======  ==========================================================
`/api/v0/feature-extraction/ <./feature_extraction_post.html>`_                         POST    Load a dataset and initialize feature extraction
`/api/v0/feature-extraction/ <./feature_extraction_get.html>`_                          GET     List processed datasets
`/api/v0/feature-extraction/<dataset-id> <./feature_extraction_element_post.html>`_     POST    Run feature extraction on a dataset
`/api/v0/feature-extraction/<dataset-id> <./feature_extraction_element_get.html>`_      GET     Load extracted features (and obtain the processing status)
`/api/v0/feature-extraction/<dataset-id>/index <./feature_extraction_index_get.html>`_  GET     Query document index for a list of filenames
`/api/v0/feature-extraction/<dataset-id> <./feature_extraction_delete.html>`_           DELETE  Delete a processed dataset
======================================================================================  ======  ==========================================================


Latent Semantic Indexing (LSI)
------------------------------

=================================================================================  =======  =========================================================
`/api/v0/lsi/ <./lsi_post.html>`_                                                  POST     Construct the Latent Semantic Indexing (LSI) model
`/api/v0/lsi/<lsi-id> <./lsi_get.html>`_                                           GET      Show LSI model parameters
`/api/v0/lsi/<lsi-id> <./lsi_delete.html>`_                                        DELETE   Delete a LSI model
=================================================================================  =======  =========================================================

Document Categorizaiton
-----------------------

=================================================================================  =======  =========================================================
`/api/v0/categorization/ <./categorization_post.html>`_                            POST     Build the ML categorization model
`/api/v0/categorization/<model-id>/predict <./categorization_predict_post.html>`_  POST     Categorize documents using different ML algorithms
`/api/v0/categorization/<model-id>/test <./categorization_test_post.html>`_        POST     Test categorization accuracy
`/api/v0/categorization/<model-id> <./categorization_element_get.html>`_           GET      Load categorization model parameters
`/api/v0/categorization/<model-id> <./categorization_element_delete.html>`_        DELETE   Delete the categorization model
=================================================================================  =======  =========================================================

Document Clustering
-------------------

======================================================================================  ======  =========================================================
`/api/v0/clustering/k-mean <./clustering_k_mean_post.html>`_                            POST    Compute clustering (K-mean)
`/api/v0/clustering/birch <./clustering_birch_post.html>`_                              POST    Compute clustering (Birch)
`/api/v0/clustering/ward-hc <./clustering_hac_post.html>`_                              POST    Compute clustering (Ward hierarchical)
`/api/v0/clustering/dbscan <./clustering_dbscan_post.html>`_                            POST    Compute clustering (DBSCAN)
`/api/v0/clustering/<model-name>/<model-id> <./clustering_model_element_get.html>`_     POST    Compute cluster labels
`/api/v0/clustering/<model-name>/<model-id> <./clustering_model_element_delete.html>`_  DELETE  Delete a clustering model
======================================================================================  ======  =========================================================

Near Duplicate Detection
------------------------

==================================================================================  ======  =========================================================
`/api/v0/duplicate-detection/ <./duplicate_detection_post.html>`_                   POST    Compute near duplicates
`/api/v0/duplicate-detection/<model-id> <./duplicate_detection_element_get.html>`_  GET     Query duplicates
==================================================================================  ======  =========================================================

Email threading
---------------

==================================================================================  ======  =========================================================
`/api/v0/email-threading/     <./email_threading_post.html>`_                       POST    Thread emails
`/api/v0/email-threading/<model-id> <./email_threading_element_get.html>`_          GET     Get threading parameters
==================================================================================  ======  =========================================================

Metrics
-------

================================================================================== ======  =========================================================
`/api/v0/metrics/categorization <./metrics_categorization_get.html>`_              GET     Retrieve categorization metrics
`/api/v0/metrics/clustering <./metrics_clustering_get.html>`_                      GET     Retrieve clustering metrics
`/api/v0/metrics/duplicate-detection <./metrics_duplicate_detection_get.html>`_    GET     Retrieve duplicate detection metrics
================================================================================== ======  =========================================================

