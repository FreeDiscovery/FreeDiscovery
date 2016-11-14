# Predict categorization with LSI


 * **URL**: `/api/v0/lsi/<lsi-id>/predict` 
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `relevant_filenames`: [required] list of relevant filenames
    - `non_relevant_filenames`: [required] list of not relevant filenames
                    

 * **Success Response**: `HTTP 200`
    
        {"id": <str>, "recall": <float>, "precision": <float> , 
         "f1": <float>,  "roc_auc": <float>, "average_precision": <float>,
         "prediction": <list[float]> ,
         "prediction_rel": <list[float]> ,
         "prediction_nrel": <list[float]> ,
         "nearest_rel_doc": <list[int]> ,
         "nearest_nrel_doc": <list[int]> }
