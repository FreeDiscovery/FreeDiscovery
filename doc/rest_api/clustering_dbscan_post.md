
# Compute clustering (DBSCAN)

The option `use_hashing=False` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

 * **URL**: `/api/v0/clustering/dbscan`
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `parent_id`: `dataset_id` or `lsi_id`
    - `eps`: (optional) float
            The maximum distance between two samples for them to be considered
             as in the same neighborhood.
    - `min_samples`: (optional) int
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself.

 * **Success Response**: `HTTP 200`
    
        {"id": <str>}
