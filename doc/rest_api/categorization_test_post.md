# Test prediction accuracy

 * **URL**: `/api/v0/categorization/<model-id>/test`
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `ground_truth_filename`: [required] tab-delimited file name with a unique document ID followed by a 1 for relevant or 0 for non-relevant document


 * **Success Response**: `HTTP 200`
 
    Returns precision, recall, f1, average precision and ROC AUC scores as [defined in scikit learn](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
    
        {"recall": <float>, "precision": <float>,
         "f1": <float>, "roc_auc": <float>, "average_precision": <float>}
