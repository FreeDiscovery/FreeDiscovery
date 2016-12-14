# Extract duplicate detection metrics

Use metrics to assess the quality of duplicate detection, comparing groud truth labels with predicted ones.

 * **URL**: `/api/v0/metrics/duplicate-detection`
 * **Method**: `GET`,  **URL Params**: None
 * **Data Params**:
    - labels_true: [required] list of int. Ground truth labels
    - labels_pred: [required] list of int. Predicted labels
    - metrics: [required] list of str. Metrics to compute, any combination of "ratio_duplicates", "f1_same_duplicates", "mean_duplicates_count"

 * **Success Response**: `HTTP 200`

Presence of keys depend on the value of `metrics` parameter. The most detailed response (all 3 metrics are called):

 {'ratio_duplicates': <float>, 'f1_same_duplicates': <float>, 'mean_duplicates_count': <float>}
