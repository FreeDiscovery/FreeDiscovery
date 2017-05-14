# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os.path
from numpy.testing import (assert_array_less, )

import pytest

from sklearn.feature_extraction.text import TfidfVectorizer

from ..text import FeatureVectorizer
from ..lsi import _TruncatedSVD_LSI, _LSIWrapper
from ..search import Search, _SearchWrapper

from .run_suite import check_cache

basename = os.path.dirname(__file__)
cache_dir = check_cache()
data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")


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
        assert dist.argmax() == best_id
        # 2 - cosine distance should be in [0, 2]
        assert_array_less(dist, 1.001)
        assert_array_less(-1 - 1e-9, dist)


@pytest.mark.parametrize('kind,', ['regular', 'semantic'])
def test_search_wrapper(kind):
    # check for syntax errors etc in the wrapper

    fe = FeatureVectorizer(cache_dir=cache_dir)
    vect_uuid = fe.setup()
    fe.ingest(data_dir, file_pattern='.*\d.txt')

    if kind == 'semantic':
        lsi = _LSIWrapper(cache_dir=cache_dir, parent_id=vect_uuid)
        lsi.fit_transform(n_components=20)
        parent_id = lsi.mid
    else:
        parent_id = vect_uuid

    sw = _SearchWrapper(cache_dir=cache_dir, parent_id=parent_id)
    dist = sw.search("so that I can reserve a room")
    assert dist.shape == (fe.n_samples_,)
    # document 1 found by
    # grep -rn "so that I can reserve a room"
    # freediscovery/data/ds_001/raw/
    assert dist.argmax() == 1
