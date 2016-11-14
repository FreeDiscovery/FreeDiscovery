# Build the categorization ML model

The option `use_hashing=True` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

 * **URL**: `/api/v0/categorization/` 
 * **Method**: `POST` **URL Params**: None
 * **Data Params**: 
    - `dataset_id`: dataset id
    - `relevant_filenames`: [required] list of relevant filenames
    - `non_relevant_filenames`: [required] list of not relevant filenames
    - `method`: classification algorithm to use (default: LogisticRegression),
          * "LogisticRegression": [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
          * "LinearSVC": [Linear SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html),
          * "xgboost": [Gradient Boosting](https://xgboost.readthedocs.io/en/latest/model.html)
           (*Warning:* for the moment xgboost is not istalled for a direct install on Windows)
    - `cv`: binary, if true optimal parameters of the ML model are determined by cross-validation over 5 stratified K-folds (default True).
    - `training_scores`: binary, compute the efficiency scores on the training dataset (default True).

 * **Success Response**: `HTTP 200`
    
        {"id": <str>, "recall": <float>, "precision": <float>,
         "f1": <float>, "roc_auc": <float>, "average_precision": <float>}
