#  Compute clustering (K-mean)

The option `use_hashing=False` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

 * **URL**: `/api/v0/clustering/k-mean`
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `parent_id`: `dataset_id` or `lsi_id`
    - `n_clusters`: the number of clusters
 * **Success Response**: `HTTP 200`
    
        {"id": <str>}
