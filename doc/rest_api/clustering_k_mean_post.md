#  Compute clustering (K-mean)

The option `use_hashing=False` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

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
