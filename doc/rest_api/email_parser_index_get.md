# Query document index for a list of filenames

 * **URL**: `/api/v0/email-parser/<dataset-id>/index`
 * **Method**: `GET`,  **URL Params**: None
 * **Data Params**: 
    - `filenames`: [required] list of filenames

 * **Success Response**: `HTTP 200`
           
         {"index": <list<int>> }
