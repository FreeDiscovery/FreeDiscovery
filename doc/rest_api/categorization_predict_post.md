# Categorize documents

Predict document categorization with a previously trained model

 * **URL**: `/api/v0/categorization/<model-id>/predict`
 * **Method**: `GET`,  **URL Params**: None

 * **Success Response**: `HTTP 200`

        {"data" : [
                    {"internal_id": <int> , "document_id": <int>,
                     "rendition_id": <int>, "score": <float>,
                     "nn_positive": {"internal_id": <int> ,
                                     "document_id": <int>,
                                     "rendition_id": <int>,
                                     "distane": <float>
                                    },
                     "nn_negative": {...}
                    },
                    {...}
                   ]
         }
