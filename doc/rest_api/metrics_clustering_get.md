# Extract clustering metrics

Use metrics to assess the quality of clustering, comparing groud truth cluster labels with predicted ones.

 * **URL**: `/api/v0/metrics/clustering`
 * **Method**: `GET`,  **URL Params**: None
 * **Data Params**:
    - labels_true: [required] list of int. Ground truth clustering labels
    - labels_pred: [required] list of int. Predicted clustering labels
    - metrics: [required] list of str. Metrics to compute, any combination of "adjusted_rand", "adjusted_mutual_info", "v_measure"

 * **Success Response**: `HTTP 200`

        {'adjusted_rand': <float>, 'adjusted_mutual_info': <float>, 'v_measure': <float>}

Presence of keys depend on the value of `metrics` parameter.
The most detailed response (all 3 metrics are called) is shown above.
