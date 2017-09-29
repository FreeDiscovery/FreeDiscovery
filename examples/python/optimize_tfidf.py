"""
.. _optimize_tfidf_scheme_example:

Optimizing TF-IDF schemes
=========================

An example of optimizing TF-IDF weighting schemes using
5 fold cross-validation
"""
from __future__ import print_function

import os
from itertools import product

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from freediscovery.feature_weighting import SmartTfidfTransformer

rng = np.random.RandomState(34)

###############################################################################
#
# We load and vectorize 2 classes from the 20 newsgroup dataset,

newsgroups = fetch_20newsgroups(subset='train',
                                categories=['sci.space', 'comp.graphics'])
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)

###############################################################################
#
# then compute baseline categorization performance using Logistic Regression
# and the TF-IDF transfomer from scikit-learn

X_tfidf = TfidfTransformer().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, newsgroups.target,
                                                    random_state=rng)

pipe = Pipeline(steps=[('tfidf', TfidfTransformer()),
                       ('logisticregression', LogisticRegression())])

pipe.fit(X_train, y_train)
print('Baseline TF-IDF categorization accuracy: {:.3f}'
      .format(pipe.score(X_test, y_test)))

###############################################################################
#
# Next, we search, using 5 fold cross-validation, for the best TF-IDF weighting
# scheme among the 80+ combinations supported by
# :class:`~freediscovery.feature_weighting.SmartTfidfTransformer`. Two
# hyper-parameters are worth optimizing in this case,
#
# * ``weighting`` parameter that defines the TF-IDF weighting (see the
#   :ref:`tfidf_section` user manual section for more details)
# * ``norm_alpha`` is the α parameter in the pivoted normalization
#   when ``weighting=="???p"``.
#
# To reduce the search parameter space in this example, we also can exclude
# the case when either the term weighting, feature weighing or normalization is
# not used as it expected to yield worse than baseline performance. We also
# exclude the non smoothed IDF weightings (``?t?``, ``?p?``) since thay return
# NaNs when some of the document frequency is 0 (which will be the case
# during cross-validation). Finally, by noticing
# that the case ``xxxp`` with  ``norm_alpha=1.0`` corresponds to the weighing
# ``xxx`` (i.e. with pivoted normalization disabled) we can reduce the search
# space even further.

pipe = Pipeline(steps=[('tfidf', SmartTfidfTransformer()),
                       ('logisticregression', LogisticRegression())])

param_grid = {'tfidf__weighting': ["".join(el) + 'p'
                                   for el in product('labLd', 'sd',
                                                     "clu")],
              'tfidf__norm_alpha': np.linspace(0, 1, 10)}

pipe_cv = GridSearchCV(pipe,
                       param_grid=param_grid,
                       verbose=1,
                       n_jobs=(1 if os.name == 'nt' else -1),
                       cv=5)
pipe_cv.fit(X_train, y_train)
print('Best CV params: weighting={weighting}, norm_alpha={norm_alpha:.3f} '
      .format(**pipe_cv.best_estimator_.steps[0][1].get_params()))
print('Best TF-IDF categorization accuracy: {:.3f}'
      .format(pipe_cv.score(X_test, y_test)))


###############################################################################
#
# In this example, by tuning TF-IDF weighting scheme with pivoted
# normalization, we obtain a categorization accuracy score of 0.99 as compared
# to a baseline TF-IDF score of 0.973. It is also interesting to notice that
# the best weighting hyper-parameter in this case is ``lnup`` which
# corresponds to the "unique pivoted normalization" case proposed by
# Singhal *et al.* (1996), although with a different α value.
