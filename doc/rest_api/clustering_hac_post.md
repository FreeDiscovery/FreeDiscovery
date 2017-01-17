
# Compute clustering (Ward hierarchical)

The option `use_hashing=False` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

The Ward Hierarchical clustering is generally slower that K-mean, however the run time can be reduced by decreasing the following parameters,

   - `lsi_components`: the number of dimensions used for the Latent Semantic Indexing decomposition (e.g. from 150 to 50)
   - `n_neighbors`:  the number of neighbors used to construct the connectivity (e.g. from 10 to 5)

 * **URL**: `/api/v0/clustering/ward-hc`
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `parent_id`: `dataset_id` or `lsi_id`
    - `n_clusters`: the number of clusters
    - `n_neighbors` Number of neighbors for each sample, used to compute the connectivity matrix (see [AgglomerativeClustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) and [kneighbors_graph](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html)
 * **Success Response**: `HTTP 200`
    
        {"id": <str>}
