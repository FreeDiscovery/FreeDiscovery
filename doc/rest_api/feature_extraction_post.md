# Load a dataset and initialize feature extraction

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

