# Compute correspondence between id fields (flat)


At least one of the fields used for indexing must be provided, and all the rest will be computed (if available)

 * **URL**: `/api/v0/feature-extraction/<dataset-id>/idx-mapping/flat`
 * **Method**: `GET`,  **URL Params**: None
 * **Data Params**: 
    - `internal_id`:  list of internal ids
    - `document_id`:  list of external document ids
    - `rendition_id`:  list of external rendition ids
    - `file_path`:  list of filenames
    - `return_file_path`:  whether the list of file paths should be returned, default: False


 * **Success Response**: `HTTP 200`
           

     {"internal_id": <list<int>> , "document_id": <list<int>>,
      "rendition_id": <list<int>>, "file_path": <list<str>> }
