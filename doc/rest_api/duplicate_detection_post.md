# Compute duplicates

The option `use_hashing=False` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

 * **URL**: `/api/v0/duplicate-detection/`
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `parent_id`: `dataset_id` or `lsi_id`
    - `method`: str, default='simhash'
         Method used for duplicate detection. One of "simhash", "i-match"

 * **Success Response**: `HTTP 200`
    
        {"id": <str>}
