#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from unittest import SkipTest
import numpy as np
from numpy.testing import assert_equal
import pytest

from freediscovery.text import FeatureVectorizer
from .run_suite import check_cache


# adapted from https://github.com/seomoz/simhash-py/blob/master/test/test.py
jabberwocky = '''
    Twas brillig, and the slithy toves
      Did gyre and gimble in the wabe:
    All mimsy were the borogoves,
      And the mome raths outgrabe.
    "Beware the Jabberwock, my son!
      The jaws that bite, the claws that catch!
    Beware the Jubjub bird, and shun
      The frumious Bandersnatch!"
    He took his vorpal sword in hand:
      Long time the manxome foe he sought --
    So rested he by the Tumtum tree,
      And stood awhile in thought.
    And, as in uffish thought he stood,
      The Jabberwock, with eyes of flame,
    Came whiffling through the tulgey wood,
      And burbled as it came!
    One, two! One, two! And through and through
      The vorpal blade went snicker-snack!
    He left it dead, and with its head
      He went galumphing back.
    "And, has thou slain the Jabberwock?
      Come to my arms, my beamish boy!
    O frabjous day! Callooh! Callay!'
      He chortled in his joy.
    `Twas brillig, and the slithy toves
      Did gyre and gimble in the wabe;
    All mimsy were the borogoves,
      And the mome raths outgrabe.'''
jabberwocky_author = ' - Lewis Carroll (Alice in Wonderland)'


def fd_setup(**fe_options):
    basename = os.path.dirname(__file__)
    cache_dir = check_cache()
    data_dir = os.path.join(basename, "..", "data", "ds_001", "raw")
    n_features = 110000
    fe = FeatureVectorizer(cache_dir=cache_dir)
    uuid = fe.preprocess(data_dir, file_pattern='.*\d.txt',
                         n_features=n_features, use_hashing=True,
                         stop_words='english',
                         **fe_options

                         )  # TODO unused variable (overwritten on the next line)
    uuid, filenames  = fe.transform()
    return cache_dir, uuid, filenames, fe


def test_simhash():

    try:
        from simhash import num_differing_bits
    except ImportError:
        raise SkipTest
    from sklearn.feature_extraction.text import HashingVectorizer
    from freediscovery.dupdet import SimhashDuplicates

    DISTANCE = 4

    fe = HashingVectorizer(ngram_range=(4,4), analyzer='word')

    X = fe.fit_transform([jabberwocky,
                          jabberwocky + jabberwocky_author,
                          jabberwocky_author,
                          jabberwocky])

    sh = SimhashDuplicates()
    sh.fit(X)

    # make sure small changes in the text results in a small number of different bytes
    assert num_differing_bits(*sh._fit_shash[:2]) <= 3
    # different text produces a large number of different bytes
    assert num_differing_bits(*sh._fit_shash[1:3]) >= 20

    # same text produces a zero bit difference
    assert num_differing_bits(*sh._fit_shash[[0,-1]]) == 0

    simhash, cluster_id, dup_pairs = sh.query(distance=DISTANCE, blocks=42)
    assert str(dup_pairs.dtype) == 'uint64'
    assert str(cluster_id.dtype) == 'int64'
    assert str(dup_pairs.dtype) == 'uint64'

    assert simhash[0] == simhash[-1]       # duplicate documents have the same simhash
    assert cluster_id[0] == cluster_id[-1] # and belong to the same cluster

    for idx, shash in enumerate(simhash):
        if (shash == simhash).sum() == 1: # ignore duplicates
            assert sh.get_index_by_hash(shash) == idx

    for pairs in dup_pairs:
        assert num_differing_bits(*pairs) <= DISTANCE


@pytest.mark.parametrize('n_rand_lexicons,', [1, 5, 100])
def test_imatch(n_rand_lexicons):

    from sklearn.feature_extraction.text import TfidfVectorizer
    from freediscovery.dupdet import IMatchDuplicates

    DISTANCE = 4

    fe = TfidfVectorizer(ngram_range=(4,4), analyzer='word',
                         min_df=0.25, max_df=0.75)

    X = fe.fit_transform([jabberwocky,
                          jabberwocky + jabberwocky_author,
                          jabberwocky_author,
                          jabberwocky])
    #print(fe.get_feature_names())

    sh = IMatchDuplicates(n_rand_lexicons=n_rand_lexicons)
    sh.fit(X)

    assert sh.labels_.shape[0] == X.shape[0]
    assert sh.hash_.shape[0] == X.shape[0]
    assert sh.hash_is_dup_.shape[0] == X.shape[0]

    # different documents produce different hash
    assert sh.labels_[0] != sh.labels_[2]

    # same text produces same hash
    assert sh.labels_[0] == sh.labels_[-1]

    # RY: not sure what other tests could be run for I-Match


@pytest.mark.parametrize('method, options, fe_options',
        [['simhash', {'distance': 3}, {} ],
         ['simhash', {'distance': 10}, {}],
         ['i-match', {}, {}]])
def test_dup_detection(method, options, fe_options):
    if method == 'simhash':
        try:
            import simhash
        except ImportError:
            raise SkipTest
    from freediscovery.dupdet import _DuplicateDetectionWrapper
    cache_dir, uuid, filenames, fe = fd_setup(**fe_options)

    dd = _DuplicateDetectionWrapper(cache_dir=cache_dir, parent_id=uuid)
    dd.fit(method=method)
    cluster_id = dd.query(**options)
    # cannot have more cluster_id than elements in the dataset
    assert len(np.unique(cluster_id)) <= len(np.unique(filenames))



