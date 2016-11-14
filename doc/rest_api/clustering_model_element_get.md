# Compute cluster labels

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
