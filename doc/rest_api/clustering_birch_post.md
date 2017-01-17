# Compute clustering (Birch)

The option `use_hashing=False` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

 * **URL**: `/api/v0/clustering/birch`
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `parent_id`: `dataset_id` or `lsi_id`
    - `n_clusters`: the number of clusters
    - `threshold`: The radius of the subcluster obtained by merging a new sample and the closest subcluster
                   should be lesser than the threshold. Otherwise a new subcluster is started. See [sklearn.cluster.Birch](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html)
 * **Success Response**: `HTTP 200`
    
        {"id": <str>}
