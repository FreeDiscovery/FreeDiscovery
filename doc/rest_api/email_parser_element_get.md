# Load parsed emails

 * **URL**: `/api/v0/email-parser/<dataset-id>`
 * **Method**: `GET`,  **URL Params**: None

 * **Success Response**: `HTTP 200` 
           
         {"id": <str>, "data_dir": <str>, "type": <str>,
          "encoding": <str>, "filenames": [] or <list[str]>}

 * **Error Response (processing failed)**: `HTTP 500`
 
        {"message": "Processing failed, see server logs!"}
