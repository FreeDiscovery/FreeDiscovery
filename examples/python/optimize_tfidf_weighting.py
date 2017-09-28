"""
Optimizing TF-IDF weighting scheme
----------------------------------

An example of optimizing TF-IDF weighting scheme using
cross-validation
"""
from __future__ import print_function

from itertools import product

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from freediscovery.feature_weighting import SmartTfidfTransformer

rng = np.random.RandomState(34)

"""
We load and vectorize 2 classes from the 20 newsgroup dataset,
"""
newsgroups = fetch_20newsgroups(subset='train',
                                categories=['sci.space', 'comp.graphics'])
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)

"""
then compute baseline categorization performance using Logistic Regression and
the TF-IDF transfomer from scikit-learn
"""
X_tfidf = TfidfTransformer().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, newsgroups.target,
                                                    random_state=rng)

pipe = Pipeline(steps=[('tfidf', TfidfTransformer()),
                       ('logisticregression', LogisticRegression())])

pipe.fit(X_train, y_train)
print('Baseline TF-IDF categorization accuracy: {:.3f}'
      .format(pipe.score(X_test, y_test)))
"""
Next, we search with cross-validation for the best TF-IDF weighting scheme
among the 5x4x4=80 possibilities supported by the `SmartTfidfTransformer`,
"""

pipe = Pipeline(steps=[('tfidf', SmartTfidfTransformer()),
                       ('logisticregression', LogisticRegression())])

param_grid = {'tfidf__weighting': ["".join(el) + 'p'
                                   for el in product('labLd', 'sd',
                                                     "clu")],
              'tfidf__norm_alpha': np.linspace(0, 1, 10)}

pipe_cv = GridSearchCV(pipe,
                       param_grid=param_grid,
                       verbose=1,
                       n_jobs=-1,
                       cv=5)
pipe_cv.fit(X_train, y_train)
print('Best CV params: weighting={weighting}, norm_alpha={norm_alpha:.3f} '
      .format(**pipe_cv.best_estimator_.steps[0][1].get_params()))
print('Best TF-IDF categorization accuracy: {:.3f}'
      .format(pipe_cv.score(X_test, y_test)))
