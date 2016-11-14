
# Test categorization with LSI

 * **URL**: `/api/v0/lsi/<lsi-id>/test` 
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `relevant_filenames`: [required] list of relevant filenames
    - `non_relevant_filenames`: [required] list of not relevant filenames
    - `ground_truth_filename`: [required] tab-delimited file name with a unique document ID followed by a 1 for relevant or 0 for non-relevant document
   
                    

 * **Success Response**: `HTTP 200`
    
        {"id": <str>, "recall": <float>, "precision": <float> ,
         "f1": <float>}
