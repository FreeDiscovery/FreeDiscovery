#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unittest import SkipTest


def test_get_positions():
    from freediscovery.interpretation import _get_keyword_positions

    textline = "Hello world, this is Eric."
    keywords = ["world", "eric"]
    positions = _get_keyword_positions(textline, keywords)
    assert(positions == [(6, 11), (21, 25)])

    textline = "email me at eric_bass@some-email.com"
    keywords = ["eric", "bass", "eric_bass"]
    positions = _get_keyword_positions(textline, keywords)
    assert(positions == [(12, 16), (12, 21), (17, 21)])

    textline = "administrative procedures"
    keywords = ["administrative", "mini", "procedure", "pro"]
    positions = _get_keyword_positions(textline, keywords)
    assert(positions == [(0, 14), (2, 6), (15, 18), (15, 24)])


def test_overlap_deduplication():
    from freediscovery.interpretation import _keep_longest_overlapping_substrings

    positions = [(6, 11), (21, 25)]
    deduplicated_positions = _keep_longest_overlapping_substrings(positions)
    assert(deduplicated_positions == [(6, 11), (21, 25)])

    positions = [(12, 16), (12, 21), (17, 21)]
    deduplicated_positions = _keep_longest_overlapping_substrings(positions)
    assert(deduplicated_positions == [(12, 21)])

    positions = [(0, 14), (2, 6), (15, 18), (15, 24)]
    deduplicated_positions = _keep_longest_overlapping_substrings(positions)
    assert(deduplicated_positions == [(0, 14), (15, 24)])


def test_explain_categorization():
    from freediscovery.interpretation import explain_categorization, _create_random_weights

    document_text = """
    Hello world\nthis is test\n
    From:           Eric Bass </o=teneo-test/ou=first administrative group/cn=recipients/cn=bass_eric>
    Sent:           Wed Jul 12 2000 15:30:00 GMT
    CC:             Subject:        Draft Order Determined
    """
    words_weights = _create_random_weights(document_text, 0.2)

    def colormap_mock(x):
       return (1.0, 1.0, 1.0, 1.0)

    try:
        document_html = explain_categorization(words_weights, document_text, colormap=colormap_mock)
    except ImportError:
        raise SkipTest
    assert len(document_html) > 0
