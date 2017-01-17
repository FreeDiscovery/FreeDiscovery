# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os.path
from unittest import SkipTest

import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
                           assert_array_less)

import pytest

from sklearn.feature_extraction.text import TfidfVectorizer

from ..lsi import _TruncatedSVD_LSI
from ..search import Search

from .run_suite import check_cache

@pytest.mark.parametrize('kind,', ['regular', 'semantic'])
def test_search(kind):
    corpus = ["To be, or not to be; that is the question;",
              "Whether ‘tis nobler in the mind to suffer",
              "The slings and arrows of outrageous fortune,",
              "Or to take arms against a sea of troubles,",
              "And by opposing end them. To die: to sleep:",
              "Nor more; and by a sleep to say we end",
              "The heart-ache and the thousand natural shocks",
              "That flesh is heir to; ‘tis a consummation",
              "Devoutly to be wished. To die; to sleep;",
              "To sleep: perchance to dream: aye, there is the rub;",
              "For in that sleep of death what dreams may come,",
              "When we have shuffled off this mortal coil,",
              "Must give us pause: there’s the respect",
              "That makes calamity of so long life;"]

    vect = TfidfVectorizer()
    X_vect = vect.fit_transform(corpus)

    if kind == 'semantic':
        lsi = _TruncatedSVD_LSI(n_components=20)
        lsi.fit(X_vect)
        X = lsi.transform_lsi_norm(X_vect)
    else:
        lsi = None
        X = X_vect

    s = Search(vect, lsi)
    s.fit(X)

    for query, best_id in [(corpus[2], 2),
                           ('death dreams', 10)]:
        dist = s.search(query)
        assert dist.shape == (X.shape[0],)
        assert dist.argmin() == best_id
