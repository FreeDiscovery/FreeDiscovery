# Extract categorization metrics

Use metrics to assess the quality of categorization, comparing groud truth labels with predicted ones.

 * **URL**: `/api/v0/metrics/categorization`
 * **Method**: `GET`,  **URL Params**: None
 * **Data Params**:
    - y_true: [required] list of int. Ground truth labels
    - y_pred: [required] list of int. Predicted labels
    - metrics: [required] list of str. Metrics to compute, any combination of "precision", "recall", "f1", "roc_auc"

 * **Success Response**: `HTTP 200`

Presence of keys depend on the value of `metrics` parameter. The most detailed response (all 4 metrics are called):

 {'precision': <float>, 'recall': <float>, 'f1': <float>, 'roc_auc': <float>}
