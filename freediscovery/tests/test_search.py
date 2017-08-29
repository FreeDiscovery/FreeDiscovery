# -*- coding: utf-8 -*-

from numpy.testing import assert_array_less

import pytest

from sklearn.feature_extraction.text import CountVectorizer
from freediscovery.feature_weighting import SmartTfidfTransformer

from freediscovery.lsi import _TruncatedSVD_LSI
from freediscovery.search import Search


@pytest.mark.parametrize('kind,', ['regular', 'semantic'])
def test_search(kind):
    # testing that search algorithm actually works
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

    vect = CountVectorizer()
    X_tf = vect.fit_transform(corpus)
    idf = SmartTfidfTransformer('nnc')
    X_vect = idf.fit_transform(X_tf)

    if kind == 'semantic':
        lsi = _TruncatedSVD_LSI(n_components=20)
        lsi.fit(X_vect)
        X = lsi.transform_lsi_norm(X_vect)
    else:
        lsi = None
        X = X_vect

    s = Search(vect, idf, lsi)
    s.fit(X)

    for query, best_id in [(corpus[2], 2),
                           ('death dreams', 10)]:
        dist = s.search(query)
        assert dist.shape == (X.shape[0],)
        assert dist.argmax() == best_id
        # 2 - cosine distance should be in [0, 2]
        assert_array_less(dist, 1.001)
        assert_array_less(-1 - 1e-9, dist)
