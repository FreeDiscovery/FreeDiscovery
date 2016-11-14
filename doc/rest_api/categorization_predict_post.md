# Categorize documents

Predict document categorization with a previously trained model

 * **URL**: `/api/v0/categorization/<model-id>/predict`
 * **Method**: `GET`,  **URL Params**: None

 * **Success Response**: `HTTP 200`

        {"prediction": <list[int]>}
