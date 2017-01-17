# (Semantic) search

Perform document search (if `dataset_id` is provided) or a semantic search (if `lsi_id` is provided). 

 * **URL**: `/api/v0/search/` 
 * **Method**: `GET` **URL Params**: None
 * **Data Params**: 
    - `parent_id`: `dataset_id` or `lsi_id`

 * **Success Response**: `HTTP 200`
    
        {"prediction": <list[float]> }
