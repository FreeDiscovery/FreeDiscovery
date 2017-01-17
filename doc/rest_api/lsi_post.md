# Build a Latent Semantic Indexing (LSI) model 

The option `use_hashing=True` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

The recommended value for the `n_components` (dimensions of the SVD decompositions) is in the [100, 200] range.

 * **URL**: `/api/v0/lsi/` 
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `n_components`: Desired dimensionality of the output data. Must be strictly less than the number of features. 
    - `parent_id`: `dataset_id`

 * **Success Response**: `HTTP 200`
    
        {"id": <str>, "explained_variance": <float> }
