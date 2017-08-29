# -*- coding: utf-8 -*-


def binary_sensitivity_analysis(estimator, vocabulary, X_row):
    """Explain the binary categorization results

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
      the binary categorization estimator
      (must have a `decision_function` method)
    vocabulary : list [n_features]
      vocabulary (list of words or n-grams)
    X_row : sparse CSR ndarray [n_features]
      a row of the document term matrix
    """
    if X_row.ndim != 2 or X_row.shape[0] != 1:
        raise ValueError('X_row must be an 2D sparse array,'
                         'with shape (1, N) not {}'.format(X_row.shape))
    if X_row.shape[1] != len(vocabulary):
        raise ValueError(('The vocabulary length ({}) does not match '
                          'the number of features in X_row ({})')
                         .format(len(vocabulary), X_row.shape[1]))

    vocabulary_inv = {ind: key for key, ind in vocabulary.items()}

    if type(estimator).__name__ == 'LogisticRegression':
        coef_ = estimator.coef_
        if X_row.shape[1] != coef_.shape[1]:
            raise ValueError(("Coefficients size {} does "
                              "not match n_features={}")
                             .format(coef_.shape[1], X_row.shape[1]))

        indices = X_row.indices
        weights = X_row.data*estimator.coef_[0, indices]
        weights_dict = {}
        for ind, value in zip(indices, weights):
            key = vocabulary_inv[ind]
            weights_dict[key] = value
        return weights_dict
    else:
        raise NotImplementedError()
