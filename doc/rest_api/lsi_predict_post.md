# Predict categorization with LSI


 * **URL**: `/api/v0/lsi/<lsi-id>/predict` 
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `index`: [required] document indices of the training set
    - `y`: [required] target binary class relative to index
                    

 * **Success Response**: `HTTP 200`
    
        { "id": <str>, "prediction": <list[float]> ,
         "dist_p": <list[float]> ,
         "dist_n": <list[float]> ,
         "ind_p": <list[int]> ,
         "ind_n": <list[int]>
         "scores":  {"recall": <float>, "precision": <float> , 
                    "f1": <float>,  "roc_auc": <float>, "average_precision": <float>}
         }
