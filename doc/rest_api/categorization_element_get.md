# Load categorization model parameters


 * **URL**: `/api/v0/categorization/<model-id>`
 * **Method**: `GET`,  **URL Params**: None

 * **Success Response**: `HTTP 200`

        {"relevant_filenames": <list>,
         "non_relevant_filenames": <list>,
         "method": <str>, "options": <dict> }
