# Compute correspondence between id fields (nested)


At least one of the fields used for indexing must be provided, and all the rest will be computed (if available)

 * **URL**: `/api/v0/feature-extraction/<dataset-id>/id-mapping/nested`
 * **Method**: `GET`,  **URL Params**: None
 * **Data Params**: 
    - `data`:  a list of dictionaties containing some of the following fields (`internal_id`, `document_id`, `rendition_id`, `file_path`)
    - `return_file_path`:  whether the list of file paths should be returned, default: False


 * **Success Response**: `HTTP 200`
           

    {"data" : [
                {"internal_id": <int> , "document_id": <int>,
                "rendition_id": <int>, "file_path": <str> },
                {"internal_id": <int> , "document_id": <int>,
                "rendition_id": <int>, "file_path": <str> },
                [...]
               ]
     }
