# Compute email threading


 * **URL**: `/api/v0/email_threading/`
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `parent_id`: `dataset_id`

 * **Success Response**: `HTTP 200`
    
        {"id": <str>, "data": <nested dict of email threads<}
