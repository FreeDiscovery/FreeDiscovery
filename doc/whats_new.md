# Release history

## Version 1.3.1

**May 22, 2018**

### Enhancements

 * Added compatibility with scikit-learn 0.19.1
 * Possibility to specify custom model ids ([#177](https://github.com/FreeDiscovery/FreeDiscovery/pull/177))
 * Optionally prune identical cluster at different depth in BIRCH

## Version 1.3.0

**Oct 1, 2017**

### New features

 * Additional TF-IDF weighting schemes and pivoted normalization ([#164](https://github.com/FreeDiscovery/FreeDiscovery/pull/164/files))
 * Exposed the wrapper functions to visualize Birch hierarchical trees in the Python package ([#175](https://github.com/FreeDiscovery/FreeDiscovery/pull/175))
 * Better separation between the FD engine (REST API) and the FD Python package.
 * Support for both Python 2.7 and 3.5+ for the Python package. The FD engine remains Python 3.5+ only. 

### Enhancements
 
 * Improved documentation and examples
 * Added compatibility with scikit-learn 0.19.0 ([#169](https://github.com/FreeDiscovery/FreeDiscovery/pull/169/files)) which fixed several issues found in 0.18.1.

### API Changes
 * In `POST /api/v0/feature-extraction` parameters `binary`, `use_idf` and  `sublinear_tf` are replaced by a single parameters `weighting` that defines term document weighting and normalization using the [SMART notation](https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System)  ([#164](https://github.com/FreeDiscovery/FreeDiscovery/pull/164/files))

## Version 1.2.0

**Jul 10, 2017**

### New features

 * Automatic scaling of the LSA dimensionality for small dataset ([#159](https://github.com/FreeDiscovery/FreeDiscovery/pull/159))

### Enhancements

 * Backported ranking metrics from sklearn 0.19 under `freediscovery.externals` ([#156](https://github.com/FreeDiscovery/FreeDiscovery/pull/156))
 * More extensive validation of the semantic search ([#158](https://github.com/FreeDiscovery/FreeDiscovery/pull/158))
 * Fix a bug in data ingestion that did not correctly use the provided `sublinear_tf` parameter. The default value was set to `sublinear_tf=False` ([#158](https://github.com/FreeDiscovery/FreeDiscovery/pull/158))

### API Changes
 * Using English stop words by default ([#155](https://github.com/FreeDiscovery/FreeDiscovery/pull/155))

## Version 1.1.2

**Jun 2, 2017**

### New features

 * Possibility to ingest documents via HTTP and to chunk document ingestion ([#143](https://github.com/FreeDiscovery/FreeDiscovery/pull/143))
 * Truncating the hierarchical BIRCH clustering tree
 * Dowloading the document collections from CLI ([#149](https://github.com/FreeDiscovery/FreeDiscovery/pull/149))

### Enhancements / Bug fixes

 * Add `subset_document_id` parameter for categrization and semantic search ([#145](https://github.com/FreeDiscovery/FreeDiscovery/pull/145))
 * Larger number of supported dependencies ([#148](https://github.com/FreeDiscovery/FreeDiscovery/pull/148))
 * Documentation improvement ([#151](https://github.com/FreeDiscovery/FreeDiscovery/pull/151))
 * Capacity to infer the `document_id` from the file name ([#150](https://github.com/FreeDiscovery/FreeDiscovery/pull/150))

### API Changes
 * Designed an unified interface for data ingestion ([#143](https://github.com/FreeDiscovery/FreeDiscovery/pull/143))
 * Added URL endpoint with server info ([#147](https://github.com/FreeDiscovery/FreeDiscovery/pull/147))

## Version 1.0

**May 2, 2017**

### New features  

 * Ability to add / remove documents in an existing processed dataset using `/api/v0/feature-extraction/{dsid}/append` and `/api/v0/feature-extraction/{dsid}/delete` URL endpoints 
 * Pagination in search and document categorization with the `batch_id` and `batch_size` parameters.

### Enhancements

 * Better handling of data persistence, which leads to faster response time for all URL endpoints, and in particular semantic search and categorization. This breaks backward compatibility for the internal data format: datasets need to re-processed and models re-trained. 
 * Additional tests for categorization and semantic search 

### API Changes
 * The `nn_metric` parameter was renamed to `metric`; a new metric `cosine-positive` was added
 * Breaking change: by default, the `cosine` similarity score is used.
 * The `/email-parser/*` endpoints are removed and merged into the `/feature-extraction/` endpoint, thus unifying data ingestion.


## Version 0.9

**Jan 28, 2017**

### New features  

 * Support for multi-class categorization and non integer class labels (PR [#96](https://github.com/FreeDiscovery/FreeDiscovery/pull/96/files)) 
 * In the case of binary categorization, recall at 20 % of documents is computed as part of the list of default scores (PR [#106](https://github.com/FreeDiscovery/FreeDiscovery/pull/106))

### Enhancements

 * Categorization and semantic search support sorting and filtering of documents below a user provided threashold. (PR [#96](https://github.com/FreeDiscovery/FreeDiscovery/pull/96/files))
 * Categorization returns only `max_result_categories` categories with the highest score. 
 * The similarity and ML scores can now be scaled to [0, 1] range using `nn_metric` and `ml_output` input parameters (PR [#101](https://github.com/FreeDiscovery/FreeDiscovery/pull/100/files)).
 * The REST API documentation is generated automatically from the code (using an OpenAPI specification) which allows to enforce consistency between the code and the docs (PR [#85](https://github.com/FreeDiscovery/FreeDiscovery/pull/85))
 * Adapted clustering and duplicate detection API to return structured objects indexed by `document_id`( and optionally `rendering_id`)
 * Improved tests coverage and overall simplified the API


### API Changes
 
 * The following endpoints accepting a request body are modified from `GET` to `POST` method (PR [#94](https://github.com/FreeDiscovery/FreeDiscovery/pull/94)), in accordance with the HTTP/1.1 spec, section 4.3,
    - `/api/v0/metrics/categorization`
    - `/api/v0/metrics/clustering`
    - `/api/v0/metrics/duplicate-detection`
    - `/api/v0/feature-extraction/{dsid}/id-mapping/flat`
    - `/api/v0/feature-extraction/{dsid}/id-mapping/nested`
    - `/api/v0/email-parser/{dsid}/index`
 * Significant changes in the categorization REST API to accommodate for multi-class cases
 * The endpoint `/api/v0/feature-extraction/{dsid}/id-mapping/flat` is removed, while `/api/v0/feature-extraction/{dsid}/id-mapping/nested` is renamed to `/api/v0/feature-extraction/{dsid}/id-mapping`. 
 * Removed the `/categorization/<mid>/test` which is superseded by `/metrics/categorization`. 
 * The `internal_id` is no longer exposed in the public API

## Version 0.8

**Feb. 25, 2017**

### New features  

 * NearestNeighbor search now can perform categorization both with positive/negative training documents (supervised) as well as with only a list of positive documents (unsupervised) (PR [#50](https://github.com/FreeDiscovery/FreeDiscovery/pull/50))
 * Document search and semantic search (PR [#63](https://github.com/FreeDiscovery/FreeDiscovery/pull/63))


### Enhancements
 
 * In depth code reorganisation making each processing step a separate REST endpoint (PR [#57](https://github.com/FreeDiscovery/FreeDiscovery/pull/57))
 * More rubust default parameters (PR [#65](https://github.com/FreeDiscovery/FreeDiscovery/pull/65))

### API Changes
 
 * Ability to associate external `document_id`, `rendition_id` fields when ingesting documents; document categorization can now be used with these external ids. 
 * All the wrappers classes are made private in the Python API
 * The same categorization and clustering enpoints can operate ether in the document-term space or in the LSI space (PR [#57](https://github.com/FreeDiscovery/FreeDiscovery/pull/57))
