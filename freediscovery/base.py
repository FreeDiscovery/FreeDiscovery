# Authors: Roman Yurchak
#
# License: BSD 3 clause


class RankerMixin(object):
    """Mixin class for all ranking estimators in FreeDiscovery.
    A ranker is a binary classifier without a decision threshold.
    """
    # so that thing would still work with scikit learn
    _estimator_type = "ranker"

    def score(self, X, y, sample_weight=None):
        """Returns the ROC score of the prediction.
        Best possible score is 1.0 and the worst in 0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples)
            True values for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            ROC score of self.decision_function(X) wrt. y.
        """

        from .metrics import roc_auc_score
        return roc_auc_score(y, self.decision_function(X),
                             sample_weight=sample_weight,)
