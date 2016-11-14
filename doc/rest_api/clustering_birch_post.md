# Compute clustering (Birch)

The option `use_hashing=False` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

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
