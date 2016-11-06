# REST API Reference

This REST API allows to use FreeDiscovery from any supported programming language. 

Following resources are defined,

## 1. Feature extraction 
### 1.a Load a dataset and initialize feature extraction

Initialize the feature extraction on a document collection.

 * **URL**: `/api/v0/feature-extraction/`
 * **Method**: `POST`,                **URL Params**: None
 * **Data Params**: (following the [sklearn.feature_extraction.text.HashingVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) API)
    - `data_dir`: [required] relative path to the directory with the input files 
    - `n_features`: [optional] number of features (overlapping character/word n-grams that are hashed). 
                    n_features refers to the number of buckets in the hash.  The larger the number, the fewer collisions.   (default: 1100000)
    - `analyzer`: 'word', 'char', 'char_wb'
                  Whether the feature should be made of word or character n-grams.
                  Option ‘char_wb’ creates character n-grams only from text inside word boundaries.  ( default: 'word')
    - `ngram_range` : tuple (min_n, max_n), default=(1, 1)
                  The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.
  
    - `stop_words`: "english" or "None"
                    Remove stop words from the resulting tokens. Only applies for the "word" analyzer.
                    If "english", a built-in stop word list for English is used. ( default: "None")
    - `n_jobs`: The maximum number of concurrently running jobs (default: 1)
    - `norm`: The normalization to use after the feature weighting ('None', 'l1', 'l2') (default: 'None')
    - `chuck_size`: The number of documents simultaneously processed by a running job (default: 5000) 
    - `binary`: If set to 1, all non zero counts are set to 1. (default: True)
    - `use_idf`: Enable inverse-document-frequency reweighting (default: False).
    - `sublinear_tf`: Apply sublinear tf scaling, i.e. replace tf with log(1 + tf) (default: False).
    - `use_hashing`: Enable hashing. This option must be set to True for classification and set to False for clustering. (default: True) 
    - `min_df`: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is ignored when hashing is used.
    - `max_df`: When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold. This value is ignored when hashing is used.



 * **Success Response**: `HTTP 200`

        {"id": <str>, "filenames": <list[str]>  }

 * **Error Response**: `HTTP 422`
        
        {"error": "Some error message"}`

### 1.b Run feature extraction on a dataset
 * **URL**: `/api/v0/feature-extraction/<dataset-id>`
 * **Method**: `POST`,  **URL Params**: None

 * **Success Response**: `HTTP 200`  
           
         {"id": <str> }


### 1.c Load extracted features (and obtain the processing status).  

 * **URL**: `/api/v0/feature-extraction/<dataset-id>`
 * **Method**: `GET`,  **URL Params**: None

 * **Success Response**: `HTTP 202` (processing in progress) or `HTTP 200` (processing done) 
           
         {"id": <str>, "data_dir": <str>, "n_samples": <int>,
          "n_features": <int>, "analyzer": <str>, "stop_words": <str>,
          "ngram_range": <list[int]>, "n_jobs": <int>, "chunk_size": <int>,
          "n_samples_processed": <int>,
          "filenames": [] or <list[str>>}

 * **Error Response (processing failed)**: `HTTP 520`
 
        {"message": "Processing failed, see server logs!"}

### 1.d List processed datasets 
 * **URL**: `/api/v0/feature-extraction/`
 * **Method**: `GET`,  **URL Params**: None
 * **Success Response**: `HTTP 200`
          
        [
          {"id": <str>, "data_dir": <str>, "n_samples": <int>,
           "n_features": <int>, "n_jobs": <int>, "chunk_size": <int>,
           "analyzer": <str>, "stop_words": <str>,
           "ngram_range": <list[int]> }, 
            ...
        ]

### 1.e Delete a processed dataset
 * **URL**: `/api/v0/feature-extraction/<dataset-id>`
 * **Method**: `DELETE`,  **URL Params**: None
 * **Success Response**: `HTTP 200`


## 2. Latent Semantic Indexing (LSI)

### 2.a Build the LSI model 

 * **URL**: `/api/v0/lsi/` 
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `n_components`: Desired dimensionality of the output data. Must be strictly less than the number of features. 
    - `dataset_id`: dataset id

 * **Success Response**: `HTTP 200`
    
        {"id": <str>, "explained_variance": <float> }

### 2.b Predict categorization with LSI


 * **URL**: `/api/v0/lsi/<lsi-id>/predict` 
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `relevant_filenames`: [required] list of relevant filenames
    - `non_relevant_filenames`: [required] list of not relevant filenames
                    

 * **Success Response**: `HTTP 200`
    
        {"id": <str>, "recall": <float>, "precision": <float> , 
         "F1": <float>,  "roc_auc": <float>, "average_precision": <float>,
         "prediction": <list[float]> ,
         "prediction_rel": <list[float]> ,
         "prediction_nrel": <list[float]> ,
         "nearest_rel_doc": <list[int]> ,
         "nearest_nrel_doc": <list[int]> }

### 2.c Test categorization with LSI

 * **URL**: `/api/v0/lsi/<lsi-id>/test` 
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `relevant_filenames`: [required] list of relevant filenames
    - `non_relevant_filenames`: [required] list of not relevant filenames
    - `ground_truth_filename`: [required] tab-delimited file name with a unique document ID followed by a 1 for relevant or 0 for non-relevant document
   
                    

 * **Success Response**: `HTTP 200`
    
        {"id": <str>, "recall_score": <float>, "precision_score": <float> ,
         "F1_score": <float>}


## 3. Document categorization

### 3.a Build the categorization model

 * **URL**: `/api/v0/categorization/` 
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `dataset_id`: dataset id
    - `relevant_filenames`: [required] list of relevant filenames
    - `non_relevant_filenames`: [required] list of not relevant filenames
    - `method`: classification algorithm to use (default: LogisticRegression),
          * "LogisticRegression": [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
          * "LinearSVC": [Linear SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html),
          * "xgboost": [Gradient Boosting](https://xgboost.readthedocs.io/en/latest/model.html)
           (*Warning:* for the moment xgboost is not istalled for a direct install on Windows)
    - `cv`: binary, if true optimal parameters of the ML model are determined by cross-validation over 5 stratified K-folds (default True).
    - `training_scores`: binary, compute the efficiency scores on the training dataset (default True).

 * **Success Response**: `HTTP 200`
    
        {"id": <str>, "recall": <float>, "precision": <float>,
         "f1": <float>, "roc_auc": <float>, "average_precision": <float>}


### 3.b Categorize documents

Predict document categorization with a previously trained model

 * **URL**: `/api/v0/categorization/<model-id>/predict`
 * **Method**: `GET`,  **URL Params**: None

 * **Success Response**: `HTTP 200`

        {"prediction": <list[int]>}


### 3.c Test prediction accuracy

 * **URL**: `/api/v0/categorization/<model-id>/test`
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `ground_truth_filename`: [required] tab-delimited file name with a unique document ID followed by a 1 for relevant or 0 for non-relevant document


 * **Success Response**: `HTTP 200`
 
    Returns precision, recall, f1, average precision and ROC AUC scores as [defined in scikit learn](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
    
        {"recall": <float>, "precision": <float>,
         "f1": <float>, "roc_auc": <float>, "average_precision": <float>}

     
### 3.d Load categorization model parameters


 * **URL**: `/api/v0/categorization/<model-id>`
 * **Method**: `GET`,  **URL Params**: None

 * **Success Response**: `HTTP 200`

        {"relevant_filenames": <list>,
         "non_relevant_filenames": <list>,
         "method": <str>, "options": <dict> }


### 3.e Delete the categorization model
 * **URL**: `/api/v0/categorization/<model-id>`
 * **Method**: `DELETE`,  **URL Params**: None
 * **Success Response**: `HTTP 200`

## 4. Document clustering

### 4.a Compute clustering (K-mean)

 * **URL**: `/api/v0/clustering/k-mean`
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `dataset_id`: dataset id
    - `n_clusters`: the number of clusters
    - `lsi_components`: (optional) apply LSI with `lsi_components` before clustering (default None)
                        Only k-means can function without the dimentionality reduction provided by LSI, 
                        both "birch" and "ward_hc" require this option to be a positive integer. 
 * **Success Response**: `HTTP 200`
    
        {"id": <str>}

### 4.b Compute clustering (Birch)

 * **URL**: `/api/v0/clustering/birch`
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `dataset_id`: dataset id
    - `n_clusters`: the number of clusters
    - `lsi_components`: (optional) apply LSI with `lsi_components` before clustering (default None)
                        Only k-means can function without the dimentionality reduction provided by LSI, 
                        both "birch" and "ward_hc" require this option to be a positive integer. 
    - `threshold`: The radius of the subcluster obtained by merging a new sample and the closest subcluster
                   should be lesser than the threshold. Otherwise a new subcluster is started. See [sklearn.cluster.Birch](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html)
 * **Success Response**: `HTTP 200`
    
        {"id": <str>}
### 4.c Compute clustering (Ward hierarchical)

 * **URL**: `/api/v0/clustering/ward_hc`
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `dataset_id`: dataset id
    - `n_clusters`: the number of clusters
    - `lsi_components`: (optional) apply LSI with `lsi_components` before clustering (default None)
                        Only k-means can function without the dimentionality reduction provided by LSI, 
                        both "birch" and "ward_hc" require this option to be a positive integer. 
    - `n_neighbors` Number of neighbors for each sample, used to compute the connectivity matrix (see [AgglomerativeClustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) and [kneighbors_graph](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html)
 * **Success Response**: `HTTP 200`
    
        {"id": <str>}

### 4.d Load results

 * **URL**: `/api/v0/clustering/<model-name>/<model-id>` 
 * **Method**: `GET` **URL Params**: None
 * **Data Params**: 
    - `n_top_words`: keep only most relevant `n_top_words` words
    - `label_method`: str, default='centroid-frequency'
                  the method used for computing the cluster labels
 * **Success Response**: `HTTP 200`
    
        {"labels": <list[int]>, "cluster_terms": <list[str]>,
         "htree": {"n_leaves": <int>, "n_components": <int>,
                   "children": <list[int[:,2]]>}   # only for ward HC
        }
        
### 4.e Delete a clustering model

 * **URL**: `/api/v0/clustering/<model-name>/<model-id>` 
 * **Method**: `DELETE` **URL Params**: None

## 5. Duplicate detection

### 5.a Compute duplicates

 * **URL**: `/api/v0/duplicate-detection/simhash`
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `dataset_id`: dataset id
    - `method`: str, default='simhash'
         Method used for duplicate detection. One of "simhash", "i-match"

 * **Success Response**: `HTTP 200`
    
        {"id": <str>}

### 5.b Query duplicates

 * **URL**: `/api/v0/duplicate-detection/<model-id>` 
 * **Method**: `GET` **URL Params**: None
 * **Data Params**: 
    - distance : int, default=2
              Maximum number of differnet bits in the simhash (Simhash method only)
    - n_rand_lexicons : int, default=1
              number of random lexicons used for duplicate detection (I-Match method only)
    - rand_lexicon_ratio: float, default=0.7
              ratio of the vocabulary used in random lexicons (I-Match method only)


 * **Success Response**: `HTTP 200`
    
        {"simhash": <list[int]>, "cluster_id": <list[int]>, "dup_pairs": <list[list[int]]> }
