# Query document index for filenames

 * **URL**: `/api/v0/feature-extraction/<dataset-id>/index`
 * **Method**: `GET`,  **URL Params**: None
 * **Data Params**: 
    - `filenames`: [required] list of filenames

 * **Success Response**: `HTTP 200`
           
         {"index": <list<int>> }
